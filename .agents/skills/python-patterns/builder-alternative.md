# Builder: Prefer defaults and factory functions

Builder classes are often used to replace missing language features. In Python, defaulted constructors and factories are usually clearer.

## Why builders overfit in Python
```py
class CarBuilder:
    def __init__(self): ...
    def set_color(self, color: str) -> "CarBuilder": ...
    def set_engine(self, engine: str) -> "CarBuilder": ...
    def build(self) -> "Car": ...
```
Usually this reimplements what constructors already support.

## Pythonic default-based construction
```py
class Car:
    def __init__(self, color: str = "black", engine: str = "V4") -> None:
        self.color = color
        self.engine = engine

car = Car(color="red", engine="V8")
```

## Factory functions for controlled construction
```py
from typing import overload

class Car:
    def __init__(self, color: str, engine: str) -> None:
        self.color = color
        self.engine = engine

@overload
def make_car() -> Car: ...
@overload
def make_car(color: str) -> Car: ...
@overload
def make_car(color: str, engine: str) -> Car: ...

def make_car(color: str = "black", engine: str = "V4") -> Car:
    return Car(color=color, engine=engine)
```

Builder remains reasonable only when object creation requires staged mutation, heavy validation, or expensive invariants.

