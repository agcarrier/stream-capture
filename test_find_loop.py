"""python3 test_find_loop.py — checks Whisper loop detection."""
from app import find_loop


def seg(t, text):
    return {"start": t, "end": t + 5, "text": text}


normal = [seg(i * 5, "line %d" % i) for i in range(20)]
assert find_loop(normal) is None

short_repeat = normal[:5] + [seg(30 + i, "Yeah.") for i in range(4)] + normal[5:]
assert find_loop(short_repeat) is None  # a few real "Yeah."s are fine

looped = normal[:10] + [seg(50 + i * 5, "Yeah.") for i in range(40)]
assert find_loop(looped) == 10
assert find_loop(looped, after=11) == 11  # still inside the run
assert find_loop(looped, after=50) is None

print("ok")
