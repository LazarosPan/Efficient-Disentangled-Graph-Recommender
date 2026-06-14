# If/Else Replacement Patterns

Use when logic is growing because of repeated equality checks, command dispatch, or nested branching.

## 1. Replace `if/elif` chains with dispatch maps
```py
STATUS_TEXT = {
    "pending": "Processing...",
    "success": "Completed!",
    "failed": "Something went wrong.",
    "cancelled": "Order cancelled.",
}

def status_message(status: str) -> str:
    return STATUS_TEXT.get(status, "Unknown status.")
```

Behavioral dispatch keeps side effects out of conditionals:
```py
def handle_pending(data): return "Processing..."
def handle_success(data): return "Completed!"

HANDLERS = {"pending": handle_pending, "success": handle_success}

def run_status(status: str, payload):
    handler = HANDLERS.get(status, lambda *_: "Unknown")
    return handler(payload)
```

## 2. Command / event routing maps
```py
def on_created(payload): ...
def on_deleted(payload): ...

EVENT_HANDLERS = {"USER_CREATED": on_created, "USER_DELETED": on_deleted}

def dispatch_event(event: str, payload):
    handler = EVENT_HANDLERS.get(event)
    if handler is None:
        raise ValueError(f"Unknown event: {event}")
    handler(payload)
```

This pattern avoids 20+ branch drift as handlers grow.

## 3. Replace nested `if` blocks with guard clauses
```py
def validate(user) -> bool:
    if not user:
        return False
    if not user.is_active:
        return False
    if not user.is_verified:
        return False
    return True
```

## 4. Pythonic alternatives
- Use `any(...)` / `all(...)` for predicate scans instead of explicit loops.
- Use ternaries for simple one-off binary choices: `x = a if condition else b`.
- Use `items or []` over `if items:` boilerplate when sensible.

## 5. Match-case (3.10+)
Use when structural matching is clearer than repeated `elif`:
```py
match cmd:
    case ("create", user):
        return f"Creating {user}"
    case ("delete", user):
        return f"Deleting {user}"
    case _:
        return "Invalid command"
```

## Use `if` when:
- logic is small (1–3 branches),
- condition is a true predicate/range check,
- readability is best with direct branching.

