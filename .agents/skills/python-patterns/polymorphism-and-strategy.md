# Polymorphism and Strategy

Use these when behavior depends on object type or mode, not just one input value.

## 1. Type-checking branch cleanup
Avoid:
```py
if isinstance(obj, Dog):
    return "woof"
if isinstance(obj, Cat):
    return "meow"
```

Prefer:
```py
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self) -> str: ...

class Dog(Animal):
    def speak(self) -> str:
        return "woof"

class Cat(Animal):
    def speak(self) -> str:
        return "meow"

def announce(animal: Animal) -> str:
    return animal.speak()
```

## 2. Strategy-style behavior switches
```py
class PricingStrategy:
    def calculate(self, price: float) -> float: ...

class Regular(PricingStrategy):
    def calculate(self, price: float) -> float:
        return price

class Premium(PricingStrategy):
    def calculate(self, price: float) -> float:
        return price * 0.9

def apply_price(price: float, strategy: PricingStrategy) -> float:
    return strategy.calculate(price)
```

When discount rules change, add/replace a class instead of editing central branch logic.

## Don’t force over-architecture
If you have two outcomes and a single obvious branch, direct `if` is often clearer.

