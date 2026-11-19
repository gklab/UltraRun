"""Host telemetry for comparing benchmark runs."""
import ctypes
from functools import cache


@cache
def _thermal_state_reader():
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    pointer = ctypes.c_void_p
    for lookup in (objc.objc_getClass, objc.sel_registerName):
        lookup.argtypes = (ctypes.c_char_p,)
        lookup.restype = pointer
    object_message = ctypes.CFUNCTYPE(pointer, pointer, pointer)(("objc_msgSend", objc))
    integer_message = ctypes.CFUNCTYPE(ctypes.c_long, pointer, pointer)(("objc_msgSend", objc))
    process_class = objc.objc_getClass(b"NSProcessInfo")
    info = object_message(process_class, objc.sel_registerName(b"processInfo"))
    selector = objc.sel_registerName(b"thermalState")
    return lambda: integer_message(info, selector) if info else None


def thermal_state():
    """Return the sampled macOS thermal state, if available."""
    try:
        state = _thermal_state_reader()()
    except (AttributeError, OSError):
        return None
    return {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}.get(state)
