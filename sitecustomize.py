import ast
import datetime

# Map deprecated AST nodes used by older pytest versions to modern equivalents.
try:
    ast.Str = ast.Constant  # type: ignore[attr-defined]
    if not hasattr(ast.Constant, 's'):
        ast.Constant.s = property(lambda self: self.value)  # type: ignore
except Exception:
    pass

# Ensure python-dateutil uses timezone-aware epoch to avoid deprecation warnings.
try:
    import importlib

    _orig = datetime.datetime.utcfromtimestamp
    datetime.datetime.utcfromtimestamp = lambda ts=0: datetime.datetime.fromtimestamp(ts, datetime.UTC)
    try:
        tz_mod = importlib.import_module("dateutil.tz.tz")
    finally:
        datetime.datetime.utcfromtimestamp = _orig

    _epoch = datetime.datetime.fromtimestamp(0, datetime.UTC)
    tz_mod.EPOCH = _epoch
    tz_mod.EPOCHORDINAL = _epoch.toordinal()
except Exception:
    pass

