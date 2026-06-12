# Python Pattern Catalog (High-Level)

Choose by intent, then open the specific files.

## Creational
- `abstract_factory` → generic factory functions.
- `factory` → delegate object creation to dedicated creator.
- `builder` → use typed constructors/defaults/factories instead of chains when possible.
- `lazy_evaluation` → delayed/lazy values and properties.
- `pool` → reuse a fixed set of objects.
- `prototype` → use factory/callable injection over clone-heavy APIs.
- `borg` → shared-state singleton variants (avoid unless clearly required).

## Structural
- `3-tier` → separate data/business/presentation responsibilities.
- `adapter` → adapt one interface to another.
- `bridge` → split abstraction and implementation.
- `composite` → treat individual objects and compositions uniformly.
- `decorator` → add behavior via wrapping.
- `facade` → hide complex subsystem details behind one API.
- `flyweight` → prefer cached functions / lightweight constructors (`flyweight-prototype.md`) over custom `__new__` caching.
- `front_controller` → centralized request handling entry point.
- `mvc` → model/view/controller coordination.
- `proxy` → controlled intermediary forwarding.

## Behavioral
- `chain_of_responsibility` → successive handlers over a data path.
- `catalog` → dispatch to specialized methods by a constructor parameter.
- `chaining_method` → chainable method continuation style.
- `command` → defer execution as data.
- `interpreter` → evaluate expressions from a simple grammar.
- `iterator` → explicit container traversal.
- `mediator` → central mediator coordinating interactions.
- `memento` → snapshot and rollback state.
- `observer` → callback on data/attribute changes.
- `publish_subscribe` → broadcast events to listeners.
- `registry` → track available subclasses.
- `servant` → shared helper service for many classes.
- `specification` → compose business rules with boolean logic.
- `state` → explicit state transition logic.
- `strategy` → swap algorithms behind a stable interface.
- `template` → fixed workflow with pluggable extension points.
- `visitor` → apply callbacks across object collections.

## Testability / Fundamentals
- `delegation_pattern` → delegate work to another object.
- `dependency_injection` → pass collaborators explicitly for testability.

## Other Architectures
- `blackboard` → cross-subsystem knowledge sharing (non-GOF).
- `graph_search` → graph-based decision/exploration architecture.
- `hsm` → hierarchical state machine.

## Anti-Patterns (typically avoid in Python-first code)
- `singleton`  
  - Modules are singleton import boundaries.  
  - Prefer module-level state or dependency injection.
- `god_object`  
  - Too many responsibilities in one class lowers testability.
- `inheritance_overuse`  
  - Deep hierarchy costs flexibility; prefer composition/delegation.

