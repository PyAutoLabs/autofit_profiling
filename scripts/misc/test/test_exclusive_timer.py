from profiling.exclusive_timer import ExclusiveTimer, shares, within_tolerance


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_nested_time_is_charged_to_the_innermost_component():
    clock = FakeClock()
    timer = ExclusiveTimer(clock=clock)

    def inner():
        clock.t += 3.0

    def outer():
        clock.t += 1.0
        timed_inner()
        clock.t += 2.0

    timed_inner = timer.wrap("nn_training", inner)
    timer.wrap("bound_geometry", outer)()
    assert timer.totals == {"nn_training": 3.0, "bound_geometry": 3.0}
    assert timer.counts == {"nn_training": 1, "bound_geometry": 1}


def test_exception_still_closes_the_interval():
    clock = FakeClock()
    timer = ExclusiveTimer(clock=clock)

    def boom():
        clock.t += 1.0
        raise RuntimeError

    try:
        timer.wrap("likelihood", boom)()
    except RuntimeError:
        pass
    assert timer.totals == {"likelihood": 1.0}
    assert not timer._stack


def test_shares_other_is_wall_minus_instrumented_and_never_negative():
    seconds = shares({"a": 2.0, "b": 1.0}, 5.0, ("a", "b", "c"))
    assert seconds == {"a": 2.0, "b": 1.0, "c": 0.0, "other": 2.0}
    assert shares({"a": 6.0}, 5.0, ("a",))["other"] == 0.0


def test_within_tolerance():
    ok, deltas = within_tolerance({"x": 0.55, "y": 0.1}, {"x": 0.5, "y": 0.25}, 0.1)
    assert not ok
    assert round(deltas["x"], 6) == 0.05 and round(deltas["y"], 6) == -0.15
    assert within_tolerance({"x": 0.55}, {"x": 0.5}, 0.1)[0]
