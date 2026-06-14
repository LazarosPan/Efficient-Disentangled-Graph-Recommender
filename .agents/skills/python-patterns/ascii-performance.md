# ASCII Conversion Performance

Use when code converts between integer lists and strings repeatedly (or in tight loops).

## 1) Convert ints -> string (use C-backed joins)

Avoid per-iteration concatenation (`s += chr(x)`), which is quadratic.

```py
def to_text(values: list[int]) -> str:
    return "".join(map(chr, values))
```

If values are guaranteed 0..255 and you want raw bytes:

```py
import array

def to_text_fast(values: list[int]) -> str:
    return array.array("B", values).tobytes().decode("ascii")
```

`array(...).tobytes()` avoids repeated Python string growth.

## 2) Convert string -> ints (fastest reverse direction)

```py
def to_codes(text: str) -> list[int]:
    return list(text.encode("ascii"))
```

Alternative for explicit type intent:

```py
import array

def to_codes_array(text: str) -> list[int]:
    return array.array("B", text.encode("ascii")).tolist()
```

## 3) Micro-optimization rules

- Profile before changing code; optimize only the proven hot loop.
- Prefer built-ins / C-backed operations over explicit Python loops.
- If using function calls in loops, minimize overhead (cache locals, avoid lambdas).
- Keep complexity checks explicit; avoid algorithms that become quadratic (`O(n**2)`) as input grows.
- Verify changes with timing on realistic input sizes.

## 4) Minimal benchmark helper

```py
import time

def benchmark(fn, reps, arg):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(arg)
    return time.perf_counter() - t0
```

Use `reps` high enough to dominate setup overhead.

