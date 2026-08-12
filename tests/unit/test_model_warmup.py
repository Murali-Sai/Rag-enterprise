"""Warming the cross-encoder at boot, without letting it hold up the boot.

Both Cloud Run services scale to zero to stay inside the budget, so a visitor
arriving after an idle period pays the cold start. Measured against the live
deployment: the landing page took 50.9s cold and 0.10s warm, and the first
query took 23.1s against 3.4s steady-state — almost all of that 23s being the
lazy `get_reranker()` singleton loading on whichever request came first.

The warmup moves that off the user's first query. What these pin is that it
cannot make things *worse*, which is the only way this optimisation could hurt:

- it must not be awaited during startup, or the seconds move into container
  boot, where Cloud Run holds the very request that triggered the cold start
- a failure to load must not fail startup, because the lazy path still works
  and a slow first query beats a dead service
- it must stay off by default, so the test suite and local runs do not load a
  model they may never use
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.main import _warm_models, app


class TestItIsOffUnlessAskedFor:
    def test_default_is_off(self):
        """The Dockerfile turns it on, the same way it pins
        ALLOW_RUNTIME_INGEST, so the property survives a deploy that forgets a
        flag. Defaulting it on would make every test load a cross-encoder."""
        assert settings.warm_models_on_startup is False


class TestItCannotBreakStartup:
    @pytest.mark.asyncio
    async def test_a_failing_load_is_swallowed(self, monkeypatch):
        """A warmup is an optimisation, never a reason to fail startup."""
        import src.retrieval.reranker as reranker

        def explode():
            raise RuntimeError("no model for you")

        monkeypatch.setattr(reranker, "get_reranker", explode)

        await _warm_models()  # must not raise

    @pytest.mark.asyncio
    async def test_it_runs_off_the_event_loop(self, monkeypatch):
        """Loading is blocking CPU and disk work.

        Run inline it would stall every concurrent request on the same worker,
        which on a single-instance deployment is all of them.
        """
        import src.retrieval.reranker as reranker

        threads = []

        def record():
            import threading

            threads.append(threading.current_thread().name)
            return object()

        monkeypatch.setattr(reranker, "get_reranker", record)

        import threading

        main_thread = threading.current_thread().name
        await _warm_models()

        assert threads and threads[0] != main_thread

    def test_startup_does_not_wait_for_it(self, monkeypatch):
        """The landing page needs neither the model nor the index.

        Simulated with a load slow enough that awaiting it would be obvious:
        the client must come up regardless, and serve while it finishes.
        """
        monkeypatch.setattr(settings, "warm_models_on_startup", True)

        import src.retrieval.reranker as reranker

        started = threading_event()

        def slow():
            started.set()
            import time

            time.sleep(2)
            return object()

        monkeypatch.setattr(reranker, "get_reranker", slow)

        import time

        began = time.monotonic()
        with TestClient(app) as client:
            elapsed = time.monotonic() - began
            response = client.get("/health")

        assert response.status_code == 200
        # Generous: the point is that it did not wait out the 2s load.
        assert elapsed < 1.5, f"startup waited for the warmup ({elapsed:.2f}s)"


def threading_event():
    import threading

    return threading.Event()


@pytest.mark.asyncio
async def test_the_task_is_cancellable():
    """Shutdown cancels a warmup still in flight rather than hanging on it."""
    task = asyncio.create_task(asyncio.sleep(30))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
