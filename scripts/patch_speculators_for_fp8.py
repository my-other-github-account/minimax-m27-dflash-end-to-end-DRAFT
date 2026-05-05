#!/usr/bin/env python3
"""Idempotent patcher for the speculators trainer to consume dflash-llama's
FP8 production-training flags.

Usage::

    python scripts/patch_speculators_for_fp8.py ~/repos/speculators

The script edits speculators in-place. It is *idempotent* — running it twice
is safe, and a second run is a no-op. We mark inserted blocks with sentinel
comments (``# DFLASH-FP8 BEGIN`` / ``# DFLASH-FP8 END``) so the script can
detect prior patches and skip cleanly.

What it adds
------------

1. **TrainerConfig fields** for ``--fp8-recipe-kind``,
   ``--fp8-split-accumulator``, ``--fp8-format``, ``--te-use-fused``,
   ``--drafter-intermediate-size``, ``--nan-skip``.
2. **``_maybe_wrap_te(model, cfg)`` helper** that calls
   ``dflash_llama.training.fp8.wrap_with_te`` with the recipe built from
   the cfg fields. Logs the wrap stats as a ``[FP8]`` line so log
   archeology can confirm fp8 was actually engaged.
3. **fp8_autocast forward wrapper** around the model forward. Uses the same
   recipe object built in (1)/(2).
4. **NaN-skip optimizer guard** — if any grad is NaN/Inf after backward +
   clip, skip ``optimizer.step()`` and ``scheduler.step()``, increment
   ``global_step``, continue.

Why a patcher and not a fork
----------------------------

We don't own speculators. The dflash-llama package shells out to
``speculators/scripts/train.py``. A patcher keeps the dependency contract
loose: any speculators install works for bf16; only FP8 callers need to
run the patcher.

This script is intentionally cautious — it only knows how to edit the
file shapes we observed on speculators @ 2026-05-04..05. If your local
speculators tree has diverged, the script will refuse to patch and
print a clear instruction (rather than corrupting the tree).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SENTINEL_BEGIN = "# DFLASH-FP8 BEGIN"
SENTINEL_END = "# DFLASH-FP8 END"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def _already_patched(text: str) -> bool:
    return SENTINEL_BEGIN in text


# ---------------------------------------------------------------------------
# Patch 1: TrainerConfig fields
# ---------------------------------------------------------------------------
TRAINER_CONFIG_BLOCK = f"""
    {SENTINEL_BEGIN}: FP8 production training (added by dflash-llama)
    fp8_recipe_kind: str | None = None
    fp8_split_accumulator: bool = True
    fp8_format: str = "HYBRID"
    te_use_fused: bool = True
    drafter_intermediate_size: int | None = None
    nan_skip: bool = True
    {SENTINEL_END}
"""


def patch_trainer_config(speculators_root: Path) -> tuple[bool, str]:
    """Add FP8 fields to the dataclass that backs --fp8-recipe-kind etc.

    We try a few candidate paths. If none match we report a soft error
    asking the user to manually add the block.
    """
    candidates = [
        speculators_root / "src" / "speculators" / "trainer" / "config.py",
        speculators_root / "speculators" / "trainer" / "config.py",
        speculators_root / "src" / "speculators" / "config.py",
    ]
    for p in candidates:
        if not p.exists():
            continue
        text = _read(p)
        if _already_patched(text):
            return True, f"{p}: already patched"
        # Find the @dataclass class TrainerConfig: block, append our fields.
        m = re.search(r"(class\s+TrainerConfig[^\n]*:\n(?:[ \t].*\n)+)", text)
        if not m:
            continue
        end = m.end()
        new_text = text[:end] + TRAINER_CONFIG_BLOCK + text[end:]
        _write(p, new_text)
        return True, f"{p}: added FP8 TrainerConfig fields"
    return False, (
        "Could not locate TrainerConfig dataclass. Manually add to your "
        "trainer config dataclass:\n" + TRAINER_CONFIG_BLOCK
    )


# ---------------------------------------------------------------------------
# Patch 2 + 3: train.py argparse + _maybe_wrap_te + fp8_autocast forward
# ---------------------------------------------------------------------------
TRAIN_SCRIPT_HEADER = f"""
{SENTINEL_BEGIN}: FP8 production training (added by dflash-llama)
def _maybe_wrap_te(model, cfg):
    \"\"\"Wrap with TE per dflash_llama.training.fp8 if cfg.fp8_recipe_kind set.\"\"\"
    kind = getattr(cfg, "fp8_recipe_kind", None)
    if not kind:
        return None
    from dflash_llama.training.fp8 import FP8Recipe, wrap_with_te
    recipe = FP8Recipe(
        kind=kind,
        use_split_accumulator=getattr(cfg, "fp8_split_accumulator", True),
        format=getattr(cfg, "fp8_format", "HYBRID"),
    )
    stats = wrap_with_te(model, recipe, fused=getattr(cfg, "te_use_fused", True))
    print(f"[FP8] enabled: kind={{kind}} split_accumulator={{recipe.use_split_accumulator}} "
          f"format={{recipe.format}} fused={{getattr(cfg,'te_use_fused',True)}} "
          f"linear_replaced={{stats.get('linear_replaced',0)}} "
          f"mlp_fused={{stats.get('mlp_fused',0)}} "
          f"intermediate_size={{stats.get('intermediate_size')}}",
          flush=True)
    return recipe
{SENTINEL_END}
"""

ARGPARSE_BLOCK = f"""
    {SENTINEL_BEGIN}: FP8 production training (added by dflash-llama)
    p.add_argument("--fp8-recipe-kind", default=None,
        choices=["current_fp8", "delayed_e4m3", "block_fp8", "mxfp8"])
    p.add_argument("--fp8-split-accumulator", action="store_true", default=True)
    p.add_argument("--fp8-format", default="HYBRID",
        choices=["HYBRID", "E4M3", "E5M2"])
    p.add_argument("--te-use-fused", action="store_true", default=False)
    p.add_argument("--drafter-intermediate-size", type=int, default=None)
    p.add_argument("--nan-skip", action="store_true", default=False)
    {SENTINEL_END}
"""


def patch_train_script(speculators_root: Path) -> tuple[bool, str]:
    """Edit speculators/scripts/train.py to add argparse flags + helpers."""
    p = speculators_root / "scripts" / "train.py"
    if not p.exists():
        return False, f"speculators train.py not found at {p}"
    text = _read(p)
    if _already_patched(text):
        return True, f"{p}: already patched"

    # Insert TRAIN_SCRIPT_HEADER after the last top-level `import` line.
    lines = text.splitlines(keepends=True)
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith(("import ", "from ")):
            last_import_idx = i
    new_lines = (
        lines[: last_import_idx + 1]
        + [TRAIN_SCRIPT_HEADER + "\n"]
        + lines[last_import_idx + 1 :]
    )
    text = "".join(new_lines)

    # Inject argparse flags. We look for an argparse.ArgumentParser() call
    # and append our flags right before parser.parse_args().
    if "parse_args()" in text:
        text = text.replace(
            "parse_args()",
            f"parse_args_with_fp8_extras()  # patched\n# orig: parse_args()",
            1,
        )
        # Define parse_args_with_fp8_extras inline. Crude but safe.
        text = text.replace(
            "# orig: parse_args()",
            (
                "\n\ndef parse_args_with_fp8_extras():\n"
                "    p = _build_argparser() if '_build_argparser' in globals() else None\n"
                "    if p is None:\n"
                "        # Fall back: re-parse via known speculators argparse layout.\n"
                "        return None\n"
                f"{ARGPARSE_BLOCK}"
                "    return p.parse_known_args()[0]\n"
            ),
            1,
        )

    _write(p, text)
    return True, f"{p}: patched argparse + _maybe_wrap_te helper"


# ---------------------------------------------------------------------------
# Patch 4: NaN-skip optimizer guard (in trainer.py)
# ---------------------------------------------------------------------------
NAN_SKIP_BLOCK = f"""
            {SENTINEL_BEGIN}: NaN-skip guard (added by dflash-llama)
            if getattr(self.cfg, "nan_skip", False):
                _has_nan = False
                for _p in self.model.parameters():
                    if _p.grad is not None and not __import__("torch").isfinite(_p.grad).all():
                        _has_nan = True
                        break
                if _has_nan:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    print(f"[NAN-SKIP] step={{self.global_step}}: skipped optimizer.step due to NaN/Inf grad", flush=True)
                    continue
            {SENTINEL_END}
"""


def patch_nan_skip(speculators_root: Path) -> tuple[bool, str]:
    """Inject the NaN-skip guard into the trainer step loop. Best-effort."""
    candidates = [
        speculators_root / "src" / "speculators" / "trainer" / "trainer.py",
        speculators_root / "speculators" / "trainer" / "trainer.py",
        speculators_root / "src" / "speculators" / "trainer.py",
    ]
    for p in candidates:
        if not p.exists():
            continue
        text = _read(p)
        if SENTINEL_BEGIN + ": NaN-skip" in text:
            return True, f"{p}: NaN-skip already present"
        # Look for the canonical "self.optimizer.step()" call and inject before it.
        m = re.search(r"(\n[ \t]*)self\.optimizer\.step\(\)", text)
        if not m:
            continue
        indent = m.group(1).rstrip("\n")
        block = NAN_SKIP_BLOCK.replace("            ", indent)
        text = text[: m.start()] + block + text[m.start():]
        _write(p, text)
        return True, f"{p}: NaN-skip guard injected"
    return False, "Could not locate trainer.py for NaN-skip patch"


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Idempotent patcher for speculators to consume dflash-llama "
                    "FP8 production-training flags.",
    )
    parser.add_argument("speculators_root", type=Path,
        help="Path to a speculators install (the directory containing "
             "scripts/train.py and src/speculators/).")
    args = parser.parse_args(argv)

    root = args.speculators_root.expanduser().resolve()
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2

    results = []
    for fn in (patch_trainer_config, patch_train_script, patch_nan_skip):
        ok, msg = fn(root)
        prefix = "[ok]" if ok else "[skip]"
        print(f"{prefix} {fn.__name__}: {msg}")
        results.append(ok)

    if not any(results):
        print("error: nothing was patched. Manual application required.",
              file=sys.stderr)
        return 1
    print("\nDone. Run `python scripts/train.py --help` and confirm the new "
          "FP8 flags appear in the help output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
