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
    from dateutil import tz

    _epoch = datetime.datetime.fromtimestamp(0, datetime.UTC)
    tz.tz.EPOCH = _epoch
    tz.tz.EPOCHORDINAL = _epoch.toordinal()
except Exception:
    pass

