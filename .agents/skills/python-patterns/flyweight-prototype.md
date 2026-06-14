# Flyweight and Prototype Alternatives

Use when Python code is reusing object instances or cloning templates.

## Flyweight: cache by value, not by `__new__`

Anti-pattern (class-level caching in `__new__`):

```py
from typing import ClassVar

class User:
    _cache: ClassVar[dict[tuple[str, int], "User"]] = {}

    def __new__(cls, name: str, age: int):
        if (key := (name, age)) not in cls._cache:
            cls._cache[key] = super().__new__(cls)
        return cls._cache[key]
```

Problems:
- shared mutable cache makes behavior hard to control and test,
- subclassing can return unexpected types,
- object lifecycle is no longer idiomatic (`__new__` overrides surprise callers).

Prefer a factory with memoization:

```py
from functools import lru_cache

class User:
    def __init__(self, name: str, age: int): ...

@lru_cache(maxsize=512)
def get_user(name: str, age: int) -> User:
    return User(name, age)
```

`lru_cache` is explicit and local to the function.

```py
class UserFactory:
    @lru_cache
    def get_user(self, name: str, age: int) -> User:
        return User(name, age)
```

This is usually wrong: each instance gets a separate cache because `self` is part of the key. Use module-level cached functions for shared behavior.

## Prototype: pass a factory, not a `clone()` protocol

If a framework needs to create objects of unknown concrete types, prefer callable injection:

```py
from collections.abc import Callable

class GraphicTool:
    def __init__(self, graph_factory: Callable[[], object]):
        self._graph_factory = graph_factory

    def click(self):
        return self._graph_factory()
```

Use:

```py
tool = GraphicTool(graph_factory=lambda: MusicalNote(note="C5"))
tool = GraphicTool(graph_factory=MusicalNote)  # default constructor
```

Avoid forcing custom `clone()` APIs unless you truly need stateful, explicit cloning semantics.

## Guideline
- Use `lru_cache` / cached function lookups for flyweight-style reuse.
- Use constructor/factory callables for dynamic framework extension.
- If logic is small and explicit object creation is clearer, keep direct `if` / direct constructors.

