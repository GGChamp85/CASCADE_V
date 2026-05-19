"""
Smoke test for the synthesizer and mixing layer.
Doesn't need torch — tests the audio generation math.
"""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Stub torch as in the math test
_torch = types.ModuleType("torch")
_torch_backends = types.ModuleType("torch.backends")
_torch_mps = types.ModuleType("torch.backends.mps")
_torch_mps.is_available = lambda: False
_torch_cuda = types.ModuleType("torch.cuda")
_torch_cuda.is_available = lambda: False
class _Device:
    def __init__(self, name): self.name = name
_torch.device = _Device
_torch.backends = _torch_backends
_torch_backends.mps = _torch_mps
_torch.cuda = _torch_cuda
sys.modules["torch"] = _torch
sys.modules["torch.backends"] = _torch_backends
sys.modules["torch.backends.mps"] = _torch_mps
sys.modules["torch.cuda"] = _torch_cuda

# Stub soundfile (only used for save_wav, which we mock)
_sf = types.ModuleType("soundfile")
def _sf_write(path, data, sr): pass
def _sf_read(path):
    import numpy as np
    return np.zeros(22050, dtype=np.float32), 22050
_sf.write = _sf_write
_sf.read = _sf_read
sys.modules["soundfile"] = _sf

# Stub torchaudio (used by audio.py)
_ta = types.ModuleType("torchaudio")
_ta_transforms = types.ModuleType("torchaudio.transforms")
class _MelSpec:
    def __init__(self, **kw): pass
    def __call__(self, x): return x
_ta_transforms.MelSpectrogram = _MelSpec
_ta.transforms = _ta_transforms
sys.modules["torchaudio"] = _ta
sys.modules["torchaudio.transforms"] = _ta_transforms


print("=" * 70)
print("Test: Catalog synthesis & nonlinear mixing (numpy-only)")
print("=" * 70)

from cascade_v.utils.synth import make_creators, synth_stem  # noqa: E402
from cascade_v.config import N_SAMPLES, SAMPLE_RATE  # noqa: E402

# 1. Generate creators with distinct signatures
creators = make_creators(n=12, seed=42)
print(f"  generated {len(creators)} creator signatures")
print(f"    creator 0: base_freq={creators[0].base_freq_hz:.0f}Hz harm={creators[0].harmonic_richness:.2f}")
print(f"    creator 5: base_freq={creators[5].base_freq_hz:.0f}Hz harm={creators[5].harmonic_richness:.2f}")
assert creators[0].base_freq_hz != creators[1].base_freq_hz
assert creators[0].harmonic_richness != creators[1].harmonic_richness
print("  PASS\n")

# 2. Synthesize stems for the same creator — must be different audios
print("Two stems by the same creator (different categories)")
stem_a_kick = synth_stem("kick", creators[0], seed=1)
stem_a_bass = synth_stem("bass", creators[0], seed=2)
print(f"  stem A (kick): shape={stem_a_kick.shape} peak={np.max(np.abs(stem_a_kick)):.3f}")
print(f"  stem A (bass): shape={stem_a_bass.shape} peak={np.max(np.abs(stem_a_bass)):.3f}")
assert stem_a_kick.shape == (N_SAMPLES,)
assert stem_a_bass.shape == (N_SAMPLES,)
assert not np.allclose(stem_a_kick, stem_a_bass)
print("  PASS\n")

# 3. Nonlinear mix
from cascade_v.generate import nonlinear_mix  # noqa: E402
print("Nonlinear mixing")
sources = [
    synth_stem("kick", creators[0], seed=1),
    synth_stem("bass", creators[3], seed=4),
    synth_stem("pad", creators[7], seed=9),
]
weights = np.array([0.5, 0.3, 0.2])
mixed = nonlinear_mix(sources, weights, seed=0)
print(f"  mixed shape: {mixed.shape}")
print(f"  mixed peak: {np.max(np.abs(mixed)):.3f}")
print(f"  mixed energy: {np.sqrt(np.mean(mixed**2)):.4f}")
assert mixed.shape == (N_SAMPLES,)
assert np.max(np.abs(mixed)) > 0.5
assert np.max(np.abs(mixed)) < 1.01
# Output should NOT equal a pure linear sum (we apply tanh + IR convolution)
linear = sum(w * s for w, s in zip(weights, sources))
linear = linear / (np.max(np.abs(linear)) + 1e-9) * 0.85
diff_pct = float(np.mean(np.abs(mixed - linear))) / float(np.mean(np.abs(linear)) + 1e-9)
print(f"  nonlinearity (mean abs diff vs linear, normalized): {diff_pct:.2%}")
assert diff_pct > 0.1, "mix is too close to linear — nonlinearity not applied"
print("  PASS\n")

print("=" * 70)
print("SYNTHESIS LAYER OK")
print("=" * 70)
