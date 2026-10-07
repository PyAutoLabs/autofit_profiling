# hpc

Driving the RAL HPC from the laptop, and getting runs back. Copied from
`autolens_inference/hpc/`, trimmed to CPU jobs: this repo profiles PyAutoFit fits and has
no GPU submits yet.

## Layout

```
sync                    laptop-side driver (below)
sync.conf.example       template; sync.conf is gitignored
batch_cpu/
  template              the shape every CPU submit is copied from
  submit_<name>         one submit per run (none yet — they arrive in B4a)
  output/   error/      SLURM stdout/stderr (gitignored; .gitkeep tracked)
```

## Partitions

CPU jobs use `--partition=ral` (`--cpus-per-task=8`). **There is no `cpu` partition on
RAL**, and bulk CPU arrays never go on `gpu` (nor `ral,gpu`) — they starve the A100s. Check
with `sinfo` before inventing a partition name.

## Running

On the login node:

```bash
cd /mnt/ral/jnightin/autofit_profiling
source activate.sh          # shared venv + PYTHONPATH at PyAutoNerves and PyAutoFit
cd hpc/batch_cpu && sbatch submit_<name>
```

The PyAuto* libraries resolve from the shared source checkouts on `PYTHONPATH` — never
pip-install them into the venv (`HPCPullPyAuto` is the update story). `activate.sh` also
sends every cache under `/mnt/ral/jnightin/.cache`, never `$HOME` or `/tmp`, and inside a
SLURM job sets `set -eE` plus an `ERR` trap so a crashed job is recorded as FAILED rather
than `COMPLETED 0:0`.

## `hpc/sync` — the laptop-side driver

```bash
cp hpc/sync.conf.example hpc/sync.conf   # then edit; sync.conf is gitignored
```

| verb | what it does |
|------|--------------|
| `hpc/sync pull` | Download batch logs, then run outputs and results |
| `hpc/sync logs` | Batch logs only — small and fast, use mid-run |
| `hpc/sync status` | Dry run: what a pull would transfer |
| `hpc/sync submit [--cpu] <name>` | `sbatch` a submit script, from `hpc/batch_cpu/` |
| `hpc/sync jobs` / `sacct` / `cancel <id>` | `squeue` / `sacct` / `scancel` |
| `hpc/sync tail [cpu]` | Stream the newest live `.out` |
| `hpc/sync du` | Remote disk usage |
| `hpc/sync check` | Verify SSH, remote paths, `sbatch`, and the local pull root |

`submit` runs `sbatch` **from `hpc/batch_cpu/`**, because the submit scripts' `#SBATCH -o`
/ `-e` paths are relative. A pull lands in **this checkout** (`LOCAL_PULL_ROOT` defaults to
the repo root); `output/` is gitignored and a pulled result row is committed by a human
from here — a job on RAL never commits.

### There is no `push`

The RAL copy is a **git clone** at `/mnt/ral/jnightin/autofit_profiling`. Code goes to RAL
with git on the login node (`git status`, then `git pull`); `hpc/sync push` prints that
procedure and exits non-zero.
