"""reward_components.py
=======================
Reusable helper + reward functions for GRPO.

Import this **both** in your training script *and* any local notebook
to debug outputs:

    from reward_components import (
        reward_fns,    # list[callable]
        reward_weights,# same length list[float]
        batch_weighted_reward, # convenience wrapper
        score_single,  # quick one‑off helper for notebooks
    )
"""
from __future__ import annotations
import re, itertools, math, csv
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List

# ─── small cache for sentence‑transformers ─────────────────────────────
_model_cache: dict[str, SentenceTransformer] = {}

def get_model(name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]

# ─── regexes & helpers ────────────────────────────────────────────────
# Require literal “as a ” (or “as an ”) — nothing looser
RE_STORY = re.compile(
    r"""
    ^\s*
    (?:                # optional leading bullet / number / markdown
        (?:\*\*?|[-•*–]|\d+[.)])\s*
    )*
    as\ (?:a|an)\s+     # ← must be “as a ” or “as an ”
    [^,]+?              # role text up to comma or ‘i want’
    (?:,)?\s*           # optional comma
    i\ want\s+.+?       # action
    \s*,?\s*            # optional comma
    so\ that\s+.+?      # benefit
    \.?$                # optional trailing period
    """,
    re.I | re.X,
)


def _story_fraction(line: str) -> float:
    line = line.lower()
    return (("as a"   in line) +
            ("i want" in line) +
            ("so that" in line)) / 3.0

def extract_final_output(text: str) -> str:
    m = re.search(r'</think>\s*', text, flags=re.I)
    return text[m.end():].strip() if m else text.strip()

def story_lines(out: str) -> List[str]:
    return [l.strip() for l in out.splitlines() if l.strip()]

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

SEP_TOKEN = "|||||"        # five vertical bars

def split_issues(ref_block: str) -> List[str]:
    """
    Return list of issue titles from the new single-line format:
        'titleA ||||| titleB ||||| titleC'
    Assumes every row in the dataset uses this delimiter.
    """
    return [
        title.strip().strip('"').strip()       # remove outer quotes/spaces
        for title in ref_block.split(SEP_TOKEN)
        if title.strip()
    ]



# ─── atomic rewards (batch signature) ──────────────────────────────────
import string
_PUNCT_TO_STRIP = string.punctuation.replace("'", "")  # keep apostrophes

def _lines_batch(completions):
    """
    Return list[list[str]] of cleaned story lines for every completion:
      • lower-cased
      • trailing punctuation removed
      • leading/trailing whitespace already trimmed
    """
    rows = []
    for c in completions:
        raw = story_lines(extract_final_output(c))
        cleaned = [ln.lower().rstrip(_PUNCT_TO_STRIP + " ") for ln in raw]
        rows.append(cleaned)
    return rows


# 1 regex structure
def r_regex(completions, **kwargs):
    return [len([l for l in ls if RE_STORY.match(l)])/len(ls) if ls else 0.0 for ls in _lines_batch(completions)]

# 2 clause presence
def r_clause(completions, **kwargs):
    outs=[]
    for ls in _lines_batch(completions):
        outs.append(sum((("as a" in l)+("i want" in l)+("so that" in l))/3 for l in ls)/len(ls) if ls else 0.0)
    return outs

# 3 coverage

def r_coverage(completions, **kwargs):
    """
    For each completion:
        • embed reference issues and generated lines
        • greedily match each issue to its most-similar unused line
        • score = average of those best cosine similarities
    """
    refs  = kwargs.get("reference_stories", [""] * len(completions))
    outs  = []
    model = get_model()

    for comp, ref in zip(completions, refs):
        lines  = story_lines(extract_final_output(comp))
        issues = split_issues(ref)

        if not (issues and lines):
            outs.append(0.0)
            continue

        emb_i = model.encode(issues, convert_to_tensor=True)
        emb_s = model.encode(lines,  convert_to_tensor=True)

        # cosine-similarity matrix  [n_issues × n_lines]
        sim_mat = torch.stack(
            [F.cosine_similarity(ie.unsqueeze(0), emb_s).cpu()
             for ie in emb_i]
        )

        # greedy one-to-one matching
        score_sum, used_cols = 0.0, set()
        for row in range(sim_mat.size(0)):
            sims_row = sim_mat[row].clone()
            for col in used_cols:
                sims_row[col] = -1.0
            best_col = torch.argmax(sims_row).item()
            used_cols.add(best_col)
            score_sum += sims_row[best_col].item()

        outs.append(score_sum / len(issues))

    return outs


# 4 story count match

def r_count(completions, **kwargs):
    refs = kwargs.get("reference_stories", [""]*len(completions)); τ=2.0
    return [math.exp(-abs(len(story_lines(extract_final_output(c))) - len(split_issues(r)))/τ) for c, r in zip(completions, refs)]

# 5 length adequacy

def r_length(completions, **kwargs):
    """
    Parabolic length reward:
        • Compute µ = average token-count of reference issue titles
        • For each generated line with n tokens:
              diff  = |n - µ|
              score = max(0, 1 - (diff / µ)**2)
        • Return the mean score over all lines
    """
    refs = kwargs.get("reference_stories", [""] * len(completions))
    outs = []

    for comp, ref in zip(completions, refs):
        issues = split_issues(ref)
        if not issues:
            outs.append(0.0)
            continue

        # mean tokens in reference issues
        mean_len = sum(len(t.split()) for t in issues) / len(issues)

        # lines produced by the model
        lines = _lines_batch([comp])[0]   # list[str]
        if not lines:
            outs.append(0.0)
            continue

        def line_score(n):
            diff = abs(n - mean_len)
            return max(0.0, 1.0 - (diff / mean_len) ** 2)

        scores = [line_score(len(l.split())) for l in lines]
        outs.append(sum(scores) / len(scores))   # average across lines

    return outs



# 6 redundancy penalty

def p_redundancy(completions, **kwargs):
    outs=[]; model=get_model(); thresh=0.8
    for ls in _lines_batch(completions):
        if len(ls)<2: outs.append(0.0); continue
        emb=model.encode(ls,convert_to_tensor=True)
        dup=sum(cosine(emb[i],emb[j])>thresh for i,j in itertools.combinations(range(len(ls)),2))
        outs.append(-dup/len(ls))
    return outs

# 7 extraneous penalty

def p_extraneous(completions, **kwargs):
    """
    Penalty = (1 − story_mass / total_lines)
    where story_mass is the *sum* of clause fractions over all lines.
    A perfect line (all three clauses) contributes 1.0; a line with only
    “as a” contributes 0.33, etc.
    """
    outs = []
    for comp in completions:
        final = extract_final_output(comp)
        lines = [l.strip() for l in final.splitlines() if l.strip()]
        total = len(lines)
        if total == 0:
            outs.append(0.0)
            continue
        story_mass = sum(_story_fraction(l) for l in lines)
        outs.append(-(total - story_mass) / total)
    return outs


# ordered list & weights
reward_fns   = [r_regex, r_clause, r_coverage, r_count, r_length, p_redundancy, p_extraneous]
reward_weights= [0.15,    0.20,    0.30,      0.1,  0.1,   0.20,        0.15]

# ─── utilities ---------------------------------------------------------------

def batch_weighted_reward(completions: list[str], reference_stories: list[str]) -> list[float]:
    """Return list of weighted sums, 1 per completion."""
    kwargs = {"reference_stories": reference_stories}
    # collect per‑function vectors
    mats = [fn(completions, **kwargs) for fn in reward_fns]
    # transpose and weight
    outs=[]
    for i in range(len(completions)):
        total=sum(reward_weights[j]*mats[j][i] for j in range(len(reward_fns)))
        outs.append(total)
    return outs

def score_single(completion: str, reference_block: str, verbose: bool=True) -> float:
    """Convenience wrapper for ad‑hoc debugging in notebooks."""
    scores=[fn([completion], reference_stories=[reference_block])[0] for fn in reward_fns]
    if verbose:
        for name, val, w in zip([fn.__name__ for fn in reward_fns], scores, reward_weights):
            print(f"{name:15s}: {val:+.3f} * {w:.2f} = {val*w:+.3f}")
        print("-"*40)
    return sum(v*w for v, w in zip(scores, reward_weights))
