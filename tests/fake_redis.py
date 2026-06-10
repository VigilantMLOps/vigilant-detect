"""FakeRedis stubs for tests — async-compatible, no real infrastructure."""
from __future__ import annotations

import asyncio


class FakePipeline:
    def __init__(self, store: dict, sets: dict) -> None:
        self._store = store
        self._sets = sets
        self._commands: list = []

    def get(self, key: str):
        self._commands.append(("get", key))
        return self

    def set(self, key: str, value):
        self._commands.append(("set", key, value))
        return self

    def sadd(self, key: str, member: str):
        self._commands.append(("sadd", key, member))
        return self

    def sismember(self, key: str, member: str):
        self._commands.append(("sismember", key, member))
        return self

    async def execute(self) -> list:
        results = []
        for cmd in self._commands:
            if cmd[0] == "get":
                results.append(self._store.get(cmd[1]))
            elif cmd[0] == "set":
                self._store[cmd[1]] = cmd[2]
                results.append(True)
            elif cmd[0] == "sadd":
                self._sets.setdefault(cmd[1], set()).add(cmd[2])
                results.append(1)
            elif cmd[0] == "sismember":
                results.append(cmd[2] in self._sets.get(cmd[1], set()))
        return results


class FakeRedis:
    """Async-compatible in-memory Redis stub."""

    def __init__(self, initial_store: dict | None = None, initial_sets: dict | None = None) -> None:
        self._store: dict = initial_store or {}
        self._sets: dict = initial_sets or {}
        self._raise_on_execute: Exception | None = None

    def set_failure(self, exc: Exception) -> None:
        """Configure to raise exc on next pipeline execute."""
        self._raise_on_execute = exc

    def pipeline(self) -> FakePipeline:
        if self._raise_on_execute is not None:
            raise self._raise_on_execute

        class FailingPipeline(FakePipeline):
            def __init__(self_, *a, **kw):
                super().__init__(*a, **kw)
                self_._exc = self._raise_on_execute

            async def execute(self_):
                if self_._exc:
                    e = self_._exc
                    self._raise_on_execute = None
                    raise e
                return await super().execute()

        return FailingPipeline(self._store, self._sets)

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass

    def seed_user(self, uid: str, last_ts: str, last_loc: str, devices: list[str]) -> None:
        """Pre-populate Redis with a known user's state."""
        self._store[f"ato:{uid}:last_ts"] = last_ts.encode()
        self._store[f"ato:{uid}:last_loc"] = last_loc.encode()
        for dev in devices:
            self._sets.setdefault(f"ato:{uid}:devices", set()).add(dev)
