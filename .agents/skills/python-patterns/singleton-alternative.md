# Singleton: Use Python modules instead

Singleton classes are usually unnecessary in Python and can hide bugs (especially with parameters and subclassing).

## Why avoid class singletons
```py
class Settings:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```
- Late calls silently mutate/observe the same object.
- Subclasses can share or collide with parent state unexpectedly.

## Pythonic replacement: module-level object
```py
class Settings:
    ...

settings = Settings()
```

For explicit single ownership across module boundaries, a module object is the singleton boundary in Python.

## Lazy creation with closure (no class singleton)
```py
from typing import Callable

def settings_cell(default: "Settings") -> tuple[Callable[[], "Settings"], Callable[["Settings"], None]]:
    _settings = default

    def get_settings() -> "Settings":
        return _settings

    def set_settings(value: "Settings") -> None:
        nonlocal _settings
        _settings = value

    return get_settings, set_settings

get_settings, set_settings = settings_cell(Settings())
```

Use lazy creation only when imports are too early or expensive.

