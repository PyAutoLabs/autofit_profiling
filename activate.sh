#!/usr/bin/env bash
# Activate the PyAuto environment for autofit_profiling: PyAutoNerves + PyAutoFit only.
#
# Two branches:
#
#   RAL     — when the shared venv exists (``$PYAUTO_HPC_BASE``, default
#             /mnt/ral/jnightin/PyAuto): activate it and point PYTHONPATH at the
#             canonical PyAuto* source checkouts beside it. pip handles only
#             third-party deps in the venv; ``HPCPullPyAuto`` is the only mechanism
#             that keeps PyAuto* current — no pip install of PyAuto* ever. This branch
#             is copied from autolens_inference/activate.sh, trimmed to the two
#             libraries this repo uses.
#   laptop  — otherwise: resolve PyAutoNerves and PyAutoFit relative to the workspace
#             root (``$PYAUTO_ROOT`` if set, else two levels above this repo, the
#             canonical ``<root>/fit/autofit_profiling`` layout) and send numba/matplotlib caches
#             to writable /tmp dirs.
#
# Usage (inside a SLURM submit or interactive shell):
#
#     source /mnt/ral/jnightin/autofit_profiling/activate.sh

# --- SLURM exit-code guard ------------------------------------------------
# Inside a batch job, abort on the first failing command so the job's exit
# status is the failure's, not the last `date`'s.
#
# WHY (RAL job 341988, 2026-08-29). Every hpc/batch_* submit ends
#
#     python3 scripts/... ; echo "Finished." ; date
#
# so the script's exit status is `date`'s and is ALWAYS 0. The W6 delaunay
# n_batch tail died on a cuFFT batched plan wanting 25.31 GiB of scratch
# (`JaxRuntimeError`), ran straight on through those two lines, and SLURM
# recorded both array tasks as COMPLETED 0:0 in 1:02 and 1:07. `sacct` showed a
# clean fast run; the traceback existed only in the .err file, and the only
# other tell was the missing results JSON. That is a failure that reports
# success, and it must not be discoverable only by noticing an absence.
#
# `set -e` is scoped to a SLURM job on purpose: this file is also sourced in
# interactive login-node shells, where errexit would close the terminal on the
# first typo. `pipefail` is deliberately NOT set — the submits pipe nothing
# into python, so pipefail would add risk without covering the failure this
# guard is for. Commands whose non-zero exit is expected already guard
# themselves with `|| echo ...` (run_probe, run_cell), which errexit leaves
# alone.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    set -eE
    trap 'rc=$?; echo "FATAL: command exited ${rc} (line ${LINENO}) — failing the SLURM job rather than reporting COMPLETED 0:0." >&2; exit ${rc}' ERR
    # A SLURM `.out` is a file, not a tty, so Python block-buffers stdout at
    # 8 KiB. RAL 341908_5 (`slam_source_pix_nn`) ran for six hours making
    # 90,000 likelihood calls and its `.out` never advanced past
    # `Calls | 0`: every progress line the driver printed was sitting in an
    # unflushed buffer that the wall-clock kill then discarded. The job was
    # therefore read as "0 calls in 6 h / it thrashes" for two days, and the
    # real failure (a likelihood-overflow flood — DECISIONS.md 2026-08-29)
    # was only found in `checkpoint.hdf5`. Line buffering costs nothing at
    # these cadences and is the difference between a live job you can watch
    # and one you can only autopsy. Scoped to SLURM alongside `set -eE` for
    # the same reason: interactive login shells are not the failure mode.
    export PYTHONUNBUFFERED=1
fi

BASE="${PYAUTO_HPC_BASE:-/mnt/ral/jnightin/PyAuto}"

if [ -f "$BASE/PyAuto/bin/activate" ]; then
    # --- RAL: shared venv + canonical PyAuto* checkouts ---------------------------

    source "$BASE/PyAuto/bin/activate"

    # --- Keep caches OFF $HOME on HPC -------------------------------------------
    # On RAL every node's /home sits on a small root disk (as does /tmp), so tools that
    # default to ~/.cache — the PyAutoNerves JAX compile cache (~/.cache/pyauto_jax), pip,
    # matplotlib, numba, CUDA/Triton kernels — fill it and break the node (RAL admin,
    # 2026-09-25). Send them to the shared project filesystem instead: the cache root sits
    # next to the PyAuto base, i.e. /mnt/ral/jnightin/.cache. Only unset variables are
    # filled, so a submit script's own JAX_COMPILATION_CACHE_DIR still wins (and an
    # explicitly EMPTY one still disables the JAX cache). The `|| true` keeps a failed
    # mkdir from tripping the SLURM `set -eE` guard above.
    export PYAUTO_HPC_CACHE="${PYAUTO_HPC_CACHE:-$(dirname "${PYAUTO_HPC_BASE:-$BASE}")/.cache}"
    mkdir -p "$PYAUTO_HPC_CACHE" 2>/dev/null || true
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PYAUTO_HPC_CACHE}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PYAUTO_HPC_CACHE/pip}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-$PYAUTO_HPC_CACHE/matplotlib}"
    export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PYAUTO_HPC_CACHE/numba}"
    export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$PYAUTO_HPC_CACHE/nv}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PYAUTO_HPC_CACHE/triton}"
    export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR-$PYAUTO_HPC_CACHE/pyauto_jax}"
    export ASTROPY_CACHE_DIR="${ASTROPY_CACHE_DIR:-$PYAUTO_HPC_CACHE/astropy}"

    export PYTHONPATH="$BASE:$BASE/PyAutoNerves:$BASE/PyAutoFit${PYTHONPATH:+:$PYTHONPATH}"
else
    # --- Laptop / workspace: sibling source checkouts -----------------------------
    _AF_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # The workspace root: $PYAUTO_ROOT if set; else the parent, when the libraries sit
    # beside this repo (a flat task bundle); else two levels up (<root>/fit/<this repo>).
    if [ -n "${PYAUTO_ROOT:-}" ]; then
        _AF_ROOT="$PYAUTO_ROOT"
    elif [ -d "$_AF_REPO/../PyAutoFit" ]; then
        _AF_ROOT="$(cd "$_AF_REPO/.." && pwd)"
    else
        _AF_ROOT="$(cd "$_AF_REPO/../.." && pwd)"
    fi
    _AF_PATH=""
    # Canonical layout (organs/PyAutoNerves, fit/PyAutoFit), else the flat bundle layout.
    for _af_pair in "organs/PyAutoNerves PyAutoNerves" "fit/PyAutoFit PyAutoFit"; do
        if [ -d "$_AF_ROOT/${_af_pair% *}" ]; then
            _AF_PATH="${_AF_PATH:+$_AF_PATH:}$_AF_ROOT/${_af_pair% *}"
        elif [ -d "$_AF_ROOT/${_af_pair#* }" ]; then
            _AF_PATH="${_AF_PATH:+$_AF_PATH:}$_AF_ROOT/${_af_pair#* }"
        fi
    done
    export PYTHONPATH="${_AF_PATH}${PYTHONPATH:+:$PYTHONPATH}"
    export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
    mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" 2>/dev/null || true
    unset _AF_REPO _AF_ROOT _AF_PATH _af_pair
fi
