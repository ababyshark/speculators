"""Client-side load balancing across multiple vLLM extractor endpoints.

The router is created once per process (each DataLoader worker gets its own
copy via spawn) and hands out one healthy endpoint per generation attempt.
It carries no cross-process or cross-node state: load balancing is achieved
by the sheer number of independently-cursored workers, and short-term skew
is absorbed by vLLM's own request queueing.

Retry semantics live with the caller (:mod:`speculators.train.data`); the
router only tracks per-endpoint health and must be told about every outcome
via :meth:`report_success` / :meth:`report_failure`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import openai

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("speculators")

DEFAULT_FAILURE_THRESHOLD = 2
DEFAULT_COOLDOWN_SECONDS = 60.0
DEFAULT_MAX_COOLDOWN_SECONDS = 600.0


def _default_client_factory(endpoint: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0)


def _dedupe(endpoints: list[str]) -> list[str]:
    """Drop duplicates while preserving order."""
    seen: set[str] = set()
    unique = []
    for endpoint in endpoints:
        if endpoint not in seen:
            seen.add(endpoint)
            unique.append(endpoint)
    return unique


@dataclass
class _EndpointState:
    """Per-endpoint health tracking (one instance per endpoint)."""

    client: Any = None
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    cooldown_seconds: float = 0.0
    evicted: bool = False
    requests: int = 0
    failures: int = 0
    recoveries: int = 0
    in_cooldown_logged: bool = field(default=False, repr=False)

    def available(self, now: float) -> bool:
        return not self.evicted and self.cooldown_until <= now

    def tripped(self) -> bool:
        """Whether the endpoint is currently serving a cooldown window."""
        return self.cooldown_seconds > 0


class EndpointRouter:
    """Round-robin dispatcher over vLLM endpoints with failure eviction.

    State machine per endpoint::

        HEALTHY ──consecutive failures >= threshold──▶ COOLDOWN(t)
           ▲                                              │ cooldown expires
           │                                          HALF-OPEN (next pick
           └────────────── success ◀── success ──── is the probe)
                                                          │ probe fails
                                                          ▼
                                                    COOLDOWN(2t, capped)

    When every endpoint is in cooldown, ``pick`` falls back to the endpoint
    with the earliest cooldown expiry so generation degrades to the
    single-endpoint retry behavior instead of raising.
    """

    def __init__(
        self,
        endpoints: list[str],
        *,
        client_factory: Callable[[str], Any] = _default_client_factory,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        max_cooldown_seconds: float = DEFAULT_MAX_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.endpoints = _dedupe(list(endpoints))
        if not self.endpoints:
            raise ValueError("EndpointRouter requires at least one endpoint")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_seconds <= 0 or max_cooldown_seconds < cooldown_seconds:
            raise ValueError(
                "cooldown_seconds must be > 0 and max_cooldown_seconds "
                f">= cooldown_seconds, got {cooldown_seconds}/{max_cooldown_seconds}"
            )
        self._states = {ep: _EndpointState() for ep in self.endpoints}
        self._cursor = 0
        self._client_factory = client_factory
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_cooldown_seconds = max_cooldown_seconds
        self._clock = clock
        self._all_down_logged = False

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def client_for(self, endpoint: str) -> Any:
        """Build (once) and return the OpenAI client for *endpoint*."""
        state = self._states[endpoint]
        if state.client is None:
            state.client = self._client_factory(endpoint)
        return state.client

    def pick(self) -> tuple[Any, str]:
        """Return ``(client, endpoint)`` for the next generation attempt."""
        now = self._clock()
        candidates = [ep for ep in self.endpoints if self._states[ep].available(now)]
        if candidates:
            endpoint = self._next_round_robin(candidates)
            self._all_down_logged = False
        else:
            alive = [ep for ep in self.endpoints if not self._states[ep].evicted]
            if not alive:
                raise RuntimeError(
                    "All vLLM endpoints have been permanently evicted "
                    f"({', '.join(self.endpoints)}); cannot route generation"
                )
            endpoint = min(alive, key=lambda ep: self._states[ep].cooldown_until)
            if not self._all_down_logged:
                logger.warning(
                    "All vLLM endpoints are unhealthy or in cooldown; "
                    "falling back to earliest-recovering endpoint %s",
                    endpoint,
                )
                self._all_down_logged = True
        return self.client_for(endpoint), endpoint

    def _next_round_robin(self, candidates: list[str]) -> str:
        candidate_set = set(candidates)
        for offset in range(len(self.endpoints)):
            index = (self._cursor + offset) % len(self.endpoints)
            endpoint = self.endpoints[index]
            if endpoint in candidate_set:
                self._cursor = (index + 1) % len(self.endpoints)
                return endpoint
        raise RuntimeError("No candidate endpoint found")  # pragma: no cover

    # ------------------------------------------------------------------
    # Outcome reporting
    # ------------------------------------------------------------------

    def report_success(self, endpoint: str) -> None:
        state = self._states[endpoint]
        state.requests += 1
        if state.tripped() and state.cooldown_until <= self._clock():
            state.recoveries += 1
            logger.info(
                "vLLM endpoint %s recovered (total recoveries: %d)",
                endpoint,
                state.recoveries,
            )
        state.consecutive_failures = 0
        state.cooldown_seconds = 0.0
        state.cooldown_until = 0.0

    def report_failure(self, endpoint: str) -> None:
        state = self._states[endpoint]
        state.requests += 1
        state.failures += 1
        if state.evicted:
            return
        now = self._clock()
        half_open = state.tripped() and state.cooldown_until <= now
        state.consecutive_failures += 1
        if not half_open and state.consecutive_failures < self.failure_threshold:
            return

        duration = (
            self.cooldown_seconds
            if state.cooldown_seconds == 0.0
            else min(state.cooldown_seconds * 2, self.max_cooldown_seconds)
        )
        state.cooldown_seconds = duration
        state.cooldown_until = now + duration
        state.consecutive_failures = 0
        state.in_cooldown_logged = True
        logger.warning(
            "vLLM endpoint %s evicted for %.0fs after %d failure(s) "
            "(total failures: %d)",
            endpoint,
            duration,
            self.failure_threshold if not half_open else 1,
            state.failures,
        )

    def evict(self, endpoint: str) -> None:
        """Permanently remove an endpoint (config error, wrong model, ...)."""
        state = self._states[endpoint]
        if state.evicted:
            return
        state.evicted = True
        logger.error("vLLM endpoint %s permanently evicted from routing", endpoint)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_healthy(self, endpoint: str) -> bool:
        state = self._states[endpoint]
        return state.available(self._clock())

    def stats(self, endpoint: str) -> dict[str, int | bool | float]:
        state = self._states[endpoint]
        return {
            "requests": state.requests,
            "failures": state.failures,
            "recoveries": state.recoveries,
            "evicted": state.evicted,
            "in_cooldown": state.tripped() and state.cooldown_until > self._clock(),
        }

    # ------------------------------------------------------------------
    # Pickling (DataLoader spawn workers must not carry live HTTP clients)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        states = {
            endpoint: vars(endpoint_state).copy()
            for endpoint, endpoint_state in self._states.items()
        }
        for endpoint_state in states.values():
            endpoint_state["client"] = None
        state["_states"] = states
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._states = {
            endpoint: _EndpointState(**endpoint_state)
            for endpoint, endpoint_state in self._states.items()
        }
