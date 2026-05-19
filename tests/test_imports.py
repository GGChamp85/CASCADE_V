"""
Final import test — validates every module in the package compiles and
imports cleanly with stubbed external dependencies.
"""

import sys
import types
from pathlib import Path

# Stubs ---------------------------------------------------------------
def make_stubs():
    # torch
    t = types.ModuleType("torch")
    tb = types.ModuleType("torch.backends")
    tm = types.ModuleType("torch.backends.mps")
    tm.is_available = lambda: False
    tc = types.ModuleType("torch.cuda")
    tc.is_available = lambda: False
    class _D:
        def __init__(self, n): self.n = n
    t.device = _D
    t.backends = tb
    tb.mps = tm
    t.cuda = tc

    # Tensor stubs
    class _Tensor: pass
    t.Tensor = _Tensor
    def _from_numpy(x): return x
    t.from_numpy = _from_numpy
    def _stack(xs): return xs
    t.stack = _stack
    def _cat(xs, **kw): return xs
    t.cat = _cat
    def _arange(n, **kw): return list(range(n))
    t.arange = _arange
    def _eye(n, **kw): return None
    t.eye = _eye
    t.bool = "bool"

    def _no_grad(): 
        class _NoGrad:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def __call__(self, fn): return fn
        return _NoGrad()
    t.no_grad = _no_grad

    def _save(*a, **kw): pass
    def _load(*a, **kw): return {}
    t.save = _save
    t.load = _load
    t.manual_seed = lambda *a: None
    
    # nn
    nn = types.ModuleType("torch.nn")
    class _Module:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return None
        def to(self, *a, **kw): return self
        def eval(self): return self
        def train(self): return self
        def parameters(self): return []
        def state_dict(self): return {}
        def load_state_dict(self, *a, **kw): pass
    nn.Module = _Module
    nn.Sequential = _Module
    nn.Linear = _Module
    nn.Conv2d = _Module
    nn.BatchNorm2d = _Module
    nn.ReLU = _Module
    nn.MaxPool2d = _Module
    nn.AdaptiveAvgPool2d = _Module
    
    # nn.functional
    F = types.ModuleType("torch.nn.functional")
    F.normalize = lambda x, **kw: x
    F.cross_entropy = lambda *a, **kw: None
    nn.functional = F
    t.nn = nn
    
    # optim
    optim = types.ModuleType("torch.optim")
    optim.AdamW = _Module
    lr_sched = types.ModuleType("torch.optim.lr_scheduler")
    class _Sch(_Module):
        def step(self): pass
        def get_last_lr(self): return [1e-3]
    lr_sched.CosineAnnealingLR = _Sch
    optim.lr_scheduler = lr_sched
    t.optim = optim
    
    # data
    data = types.ModuleType("torch.utils.data")
    data.Dataset = _Module
    data.DataLoader = _Module
    utils = types.ModuleType("torch.utils")
    utils.data = data
    t.utils = utils
    
    sys.modules.update({
        "torch": t,
        "torch.backends": tb,
        "torch.backends.mps": tm,
        "torch.cuda": tc,
        "torch.nn": nn,
        "torch.nn.functional": F,
        "torch.optim": optim,
        "torch.optim.lr_scheduler": lr_sched,
        "torch.utils": utils,
        "torch.utils.data": data,
    })
    
    # torchaudio
    ta = types.ModuleType("torchaudio")
    tat = types.ModuleType("torchaudio.transforms")
    class _Mel:
        def __init__(self, **kw): pass
        def __call__(self, x): return x
    tat.MelSpectrogram = _Mel
    ta.transforms = tat
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.transforms"] = tat
    
    # soundfile
    sf = types.ModuleType("soundfile")
    sf.write = lambda *a, **kw: None
    sf.read = lambda p: ([0.0], 22050)
    sys.modules["soundfile"] = sf

    # z3 — needs to support arithmetic on Real expressions
    z3 = types.ModuleType("z3")
    class _Expr:
        def __init__(self, name="?"): self.name = name
        def __sub__(self, o): return _Expr(f"({self.name} - {o})")
        def __rsub__(self, o): return _Expr(f"({o} - {self.name})")
        def __add__(self, o): return _Expr(f"({self.name} + {o})")
        def __radd__(self, o): return _Expr(f"({o} + {self.name})")
        def __lt__(self, o): return _Expr(f"({self.name} < {o})")
        def __le__(self, o): return _Expr(f"({self.name} <= {o})")
        def __gt__(self, o): return _Expr(f"({self.name} > {o})")
        def __ge__(self, o): return _Expr(f"({self.name} >= {o})")
        def __eq__(self, o): return _Expr(f"({self.name} == {o})")
        def __abs__(self): return _Expr(f"abs({self.name})")
    class _S:
        def __init__(self): self._a = []; self._snap = None
        def add(self, *a): self._a.extend(a)
        def push(self): self._snap = list(self._a)
        def pop(self):
            if self._snap is not None:
                self._a = self._snap
        def check(self): return z3.sat
    z3.Solver = _S
    z3.Real = lambda n: _Expr(n)
    def _Sum(xs):
        if len(xs) == 0: return _Expr("0")
        e = xs[0]
        for x in xs[1:]: e = e + x
        return e
    z3.Sum = _Sum
    z3.And = lambda *a: _Expr("and(" + ",".join(str(x) for x in a) + ")")
    z3.Or = lambda *a: _Expr("or(" + ",".join(str(x) for x in a) + ")")
    z3.sat = "sat"
    z3.unsat = "unsat"
    sys.modules["z3"] = z3

    # tqdm
    tq = types.ModuleType("tqdm")
    tq.tqdm = lambda x, **kw: x
    sys.modules["tqdm"] = tq

    # rich
    rich = types.ModuleType("rich")
    rc = types.ModuleType("rich.console")
    class _Console:
        def print(self, *a, **kw): pass
    rc.Console = _Console
    rt = types.ModuleType("rich.table")
    class _Table:
        def __init__(self, **kw): pass
        def add_column(self, *a, **kw): pass
        def add_row(self, *a, **kw): pass
    rt.Table = _Table
    rich.console = rc
    rich.table = rt
    sys.modules["rich"] = rich
    sys.modules["rich.console"] = rc
    sys.modules["rich.table"] = rt

    # rich.logging.RichHandler — used by cascade_v.logging_setup
    rl = types.ModuleType("rich.logging")

    class _RichHandler:
        def __init__(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
            pass

        def setLevel(self, *args, **kwargs):  # noqa: ANN001, ARG002
            pass

        def setFormatter(self, *args, **kwargs):  # noqa: ANN001, ARG002
            pass

    rl.RichHandler = _RichHandler
    sys.modules["rich.logging"] = rl
    rich.logging = rl

    # typer
    ty = types.ModuleType("typer")
    class _Typer:
        def __init__(self, **kw): pass
        def command(self):
            def _d(fn): return fn
            return _d
    ty.Typer = _Typer
    ty.Argument = lambda *a, **kw: None
    ty.Option = lambda *a, **kw: None
    class _Exit(Exception): pass
    ty.Exit = _Exit
    sys.modules["typer"] = ty

    # pydantic + pydantic-settings — used by cascade_v.settings to validate
    # the runtime configuration. The real packages are heavy and pull in
    # rust extensions, so we stub just the surface that settings.py uses.
    pd = types.ModuleType("pydantic")

    def _Field(default=None, default_factory=None, **kwargs):  # noqa: ANN001
        if default_factory is not None:
            return default_factory()
        return default

    def _field_validator(*args, **kwargs):  # noqa: ANN001, ANN002, ARG001
        def deco(fn):
            return classmethod(fn)
        return deco

    pd.Field = _Field
    pd.field_validator = _field_validator
    sys.modules["pydantic"] = pd

    pds = types.ModuleType("pydantic_settings")

    class _BaseSettings:
        # Minimal stand-in: accept kwargs, set them as attributes, fall
        # back to class-level defaults for anything not supplied. Mimics
        # just the part of pydantic-settings that settings.Settings uses.
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            # Run model_post_init if defined (used by Settings to cross-validate)
            post = getattr(self, "model_post_init", None)
            if callable(post):
                try:
                    post(None)
                except Exception:
                    pass

    class _SettingsConfigDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(**kwargs)

    pds.BaseSettings = _BaseSettings
    pds.SettingsConfigDict = _SettingsConfigDict
    sys.modules["pydantic_settings"] = pds

make_stubs()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Import everything ----------------------------------------------------
modules_to_check = [
    "cascade_v",
    "cascade_v.config",
    "cascade_v.types",
    "cascade_v.utils.audio",
    "cascade_v.utils.synth",
    "cascade_v.encoder",
    "cascade_v.train",
    "cascade_v.embeddings",
    "cascade_v.generate",
    "cascade_v.stages.stage1_triage",
    "cascade_v.stages.stage2_grouping",
    "cascade_v.stages.stage3_shapley",
    "cascade_v.verification.validators",
    "cascade_v.verification.intervals",
    "cascade_v.verification.proofs",
    "cascade_v.pipeline",
    "cascade_v.baselines",
    "cascade_v.receipts",
    "cascade_v.evaluate",
]

print("Importing every module:")
ok = 0
for m in modules_to_check:
    try:
        __import__(m)
        print(f"  OK  {m}")
        ok += 1
    except Exception as e:
        print(f"  FAIL {m}: {type(e).__name__}: {e}")

print(f"\n{ok}/{len(modules_to_check)} modules imported successfully")
assert ok == len(modules_to_check), "some modules failed to import"
print("\nAll modules compile and import cleanly.")
