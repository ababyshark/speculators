"""Unit tests for the EndpointRouter load balancer state machine."""

import pickle

import pytest

from speculators.data_generation.endpoint_router import EndpointRouter


def fake_client_factory(endpoint: str) -> str:
    """Module-level so pickled routers can rebuild clients."""
    return f"client::{endpoint}"


class FakeClock:
    """Controllable monotonic clock for cooldown tests."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _router(endpoints, **kwargs) -> tuple[EndpointRouter, FakeClock]:
    clock = FakeClock()
    router = EndpointRouter(
        endpoints,
        client_factory=fake_client_factory,
        clock=clock,
        **kwargs,
    )
    return router, clock


def _pick_all(router: EndpointRouter) -> list[str]:
    """Pick len(endpoints) times, returning the endpoint sequence."""
    return [router.pick()[1] for _ in range(len(router.endpoints))]


class TestConstruction:
    def test_empty_endpoints_raise(self):
        with pytest.raises(ValueError, match="at least one endpoint"):
            EndpointRouter([])

    def test_duplicates_dropped_preserving_order(self):
        router, _ = _router(["b", "a", "b", "c", "a"])
        assert router.endpoints == ["b", "a", "c"]

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="failure_threshold"):
            EndpointRouter(["a"], failure_threshold=0)

    def test_invalid_cooldown_raises(self):
        with pytest.raises(ValueError, match="cooldown_seconds"):
            EndpointRouter(["a"], cooldown_seconds=10, max_cooldown_seconds=5)

    def test_lazy_client_creation(self):
        created = []
        router, _ = _router(["a", "b"])
        assert all(router._states[ep].client is None for ep in router.endpoints)
        client = router.client_for("a")
        created.append(client)
        assert router.client_for("a") is created[0]


class TestRoundRobin:
    def test_alternates_between_endpoints(self):
        router, _ = _router(["a", "b"])
        picks = [router.pick()[1] for _ in range(4)]
        assert picks == ["a", "b", "a", "b"]

    def test_cursor_survives_skipped_unhealthy_endpoint(self):
        router, clock = _router(["a", "b", "c"])
        assert _pick_all(router) == ["a", "b", "c"]
        # b fails into cooldown; next picks skip it and round-robin over a/c
        router.report_failure("b")
        router.report_failure("b")
        clock.advance(1)
        assert router.pick()[1] == "a"
        assert router.pick()[1] == "c"
        assert router.pick()[1] == "a"
        assert router.pick()[1] == "c"


class TestFailureEviction:
    def test_single_failure_does_not_evict(self):
        router, _ = _router(["a", "b"])
        router.report_failure("a")
        assert router.is_healthy("a")

    def test_threshold_consecutive_failures_evict(self):
        router, clock = _router(["a", "b"], failure_threshold=2)
        router.report_failure("a")
        router.report_failure("a")
        clock.advance(1)
        assert not router.is_healthy("a")
        assert router.pick()[1] == "b"

    def test_success_resets_failure_count(self):
        router, clock = _router(["a", "b"], failure_threshold=2)
        router.report_failure("a")
        router.report_success("a")
        router.report_failure("a")
        clock.advance(1)
        assert router.is_healthy("a")

    def test_eviction_cooldown_default_duration(self):
        router, clock = _router(["a", "b"])
        router.report_failure("a")
        router.report_failure("a")
        assert not router.is_healthy("a")
        clock.advance(59.9)
        assert not router.is_healthy("a")
        clock.advance(0.2)
        assert router.is_healthy("a")


class TestHalfOpen:
    def test_half_open_failure_retrips_with_doubled_cooldown(self):
        router, clock = _router(["a", "b"], failure_threshold=2, cooldown_seconds=60)
        router.report_failure("a")
        router.report_failure("a")
        clock.advance(60)
        assert router.is_healthy("a")  # half-open: next pick is the probe
        router.report_failure("a")
        # probe failed -> immediate re-trip at 120s, not another 60s
        clock.advance(60)
        assert not router.is_healthy("a")
        clock.advance(61)
        assert router.is_healthy("a")

    def test_recovery_after_successful_probe(self):
        router, clock = _router(["a", "b"], failure_threshold=2)
        router.report_failure("a")
        router.report_failure("a")
        clock.advance(60)
        router.pick()
        router.report_success("a")
        assert router.is_healthy("a")
        stats = router.stats("a")
        assert stats["recoveries"] == 1
        assert stats["failures"] == 2


class TestExponentialBackoff:
    def test_repeated_failures_double_up_to_cap(self):
        router, clock = _router(
            ["a", "b"],
            failure_threshold=1,
            cooldown_seconds=60,
            max_cooldown_seconds=600,
        )
        # first trip at the base duration...
        router.report_failure("a")
        assert router._states["a"].cooldown_seconds == 60
        # ...then each failed half-open probe doubles the window, capped at 600
        for expected in (120, 240, 480, 600, 600):
            clock.advance(router._states["a"].cooldown_seconds)
            assert router.is_healthy("a")  # half-open
            router.report_failure("a")  # probe fails -> immediate re-trip
            assert router._states["a"].cooldown_seconds == expected


class TestAllDownFallback:
    def test_dead_end_fallback_picks_earliest_recovery(self):
        router, clock = _router(["a", "b"], failure_threshold=1, cooldown_seconds=30)
        router.report_failure("a")
        router.report_failure("b")
        clock.advance(1)
        # both down: a recovers first (tripped 1s earlier)
        client, endpoint = router.pick()
        assert endpoint == "a"

    def test_fallback_logs_warning_once_per_episode(self, caplog):
        router, clock = _router(["a", "b"], failure_threshold=1, cooldown_seconds=30)
        router.report_failure("a")
        router.report_failure("b")
        clock.advance(1)
        with caplog.at_level("WARNING", logger="speculators"):
            router.pick()
            router.pick()
        assert caplog.text.count("falling back to earliest-recovering") == 1

    def test_evicted_endpoint_never_picked_even_as_fallback(self):
        router, clock = _router(["a", "b"], failure_threshold=1, cooldown_seconds=30)
        router.evict("a")
        router.report_failure("b")
        clock.advance(1)
        for _ in range(3):
            assert router.pick()[1] == "b"


class TestEvict:
    def test_evict_is_permanent(self):
        router, clock = _router(["a", "b"])
        router.evict("a")
        clock.advance(10_000)
        picks = [router.pick()[1] for _ in range(4)]
        assert picks == ["b", "b", "b", "b"]
        assert router.stats("a")["evicted"]

    def test_all_endpoints_evicted_raises(self):
        router, clock = _router(["a", "b"])
        router.evict("a")
        router.evict("b")
        with pytest.raises(RuntimeError, match="permanently evicted"):
            router.pick()


class TestPickling:
    def test_pickle_drops_live_clients_and_keeps_counters(self):
        router, clock = _router(["a", "b"], failure_threshold=1)
        router.pick()  # force client creation for "a"
        router.report_failure("a")
        router.evict("b")

        restored = pickle.loads(pickle.dumps(router))  # noqa: S301 - trusted test data

        assert restored.endpoints == router.endpoints
        assert restored._states["a"].failures == 1
        assert restored._states["b"].evicted
        assert all(state.client is None for state in restored._states.values())
        # still functional: builds fresh clients on demand
        client, endpoint = restored.pick()
        assert endpoint == "a"
        assert client == f"client::{endpoint}"
