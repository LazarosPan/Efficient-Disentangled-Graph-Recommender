---
name: python-patterns
description: Compact Python patterns, anti-pattern alternatives, and performance notes across control flow, object creation, and structure.
---

Use this skill when deciding between branching and Pythonic alternatives.

- [pattern-catalog.md](pattern-catalog.md): high-level pattern index (creational/structural/behavioral), plus Python anti-pattern guidance.
- [if-else-replacement.md](if-else-replacement.md): replace if/elif chains with maps, callables, guards, and match-case.
- [polymorphism-and-strategy.md](polymorphism-and-strategy.md): replace type/behavior branching with polymorphic objects.
- [singleton-alternative.md](singleton-alternative.md): avoid class-based singleton anti-patterns in Python.
- [builder-alternative.md](builder-alternative.md): replace builder classes with defaults and focused factories.
- [flyweight-prototype.md](flyweight-prototype.md): prefer cached factories over `__new__` caches, and inject constructors/callables over clone-based prototype setup.
- [ascii-performance.md](ascii-performance.md): avoid slow Python-level loops by using C-backed built-ins for ASCII conversion and benchmarking.
