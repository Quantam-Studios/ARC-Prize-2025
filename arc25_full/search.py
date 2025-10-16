
from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple
from .grids import Grid, shape, histogram
from .transforms import Transform, compose, T_identity, T_rot, T_flip_h, T_flip_v, T_transpose, T_crop_nz_bbox, T_largest_component, T_smallest_component, T_scale_up, T_scale_down, T_border, T_translate, T_recolor

# A small library of search primitives with few parameter choices
def library() -> List[Transform]:
    basics = [
        T_identity(),
        T_rot(1), T_rot(2), T_rot(3),
        T_flip_h(), T_flip_v(), T_transpose(),
        T_crop_nz_bbox(),
        T_largest_component(), T_smallest_component(),
    ]
    # small set of translations and borders
    trans = [T_translate(dr,dc) for dr in (-2,-1,0,1,2) for dc in (-2,-1,0,1,2) if not (dr==0 and dc==0)]
    borders = [T_border(c,1) for c in range(1,10)]
    return basics + trans + borders

def compose_up_to_len(max_len: int) -> List[List[Transform]]:
    # return sequences of transforms of length 1..max_len
    lib = library()
    seqs: List[List[Transform]] = []
    for t in lib:
        seqs.append([t])
    if max_len >= 2:
        for a in lib:
            for b in lib:
                seqs.append([a,b])
    if max_len >= 3:
        for a in lib:
            for b in lib:
                for c in lib:
                    seqs.append([a,b,c])
    return seqs

def run_sequence(seq: List[Transform], g: Grid) -> Grid:
    out = g
    for t in seq:
        out = t.fn(out)
    return out


def pairs_key(pairs):
    """Return a simple immutable key for a given pairs list."""
    return id(pairs)

@lru_cache(maxsize=None)
def sequence_heuristic_cached(seq_tuple, pairs_id):
    """Estimate how promising a transform sequence is based on color overlap."""
    # 'pairs_id' is a dummy; used only as cache differentiator per task
    # We'll look up actual pairs from a global cache
    pairs = _pairs_lookup[pairs_id]
    score = 0
    for inp, out in pairs:
        try:
            pred = run_sequence(list(seq_tuple), inp)
            h_pred = histogram(pred)
            h_out = histogram(out)
            common = set(h_pred).intersection(h_out)
            score += sum(min(h_pred[c], h_out[c]) for c in common)
        except Exception:
            continue
    return score

# global lookup for current pairs (local to search.py)
_pairs_lookup = {}

def fit_on_pairs(pairs: List[Tuple[Grid, Grid]], max_len: int = 3, limit: int = 6000) -> List[Transform]:
    """Enumerate short programs and keep those that exactly solve all train pairs."""
    # store in lookup for access from cached function
    pairs_id = id(pairs)
    _pairs_lookup[pairs_id] = pairs

    seqs = compose_up_to_len(max_len)
    if limit is not None:
        seqs = seqs[:limit]

    # sort by heuristic score (descending)
    seqs = sorted(
        seqs,
        key=lambda s: sequence_heuristic_cached(tuple(s), pairs_id),
        reverse=True
    )

    valids: List[Transform] = []
    seen = set()

    for seq in seqs:
        ok = True
        for inp, out in pairs:
            try:
                pred = run_sequence(seq, inp)
            except Exception:
                ok = False
                break
            if pred != out:
                ok = False
                break
        if ok:
            from .transforms import compose
            tf = seq[0]
            for s in seq[1:]:
                tf = compose(tf, s)
            if tf.name not in seen:
                valids.append(tf)
                seen.add(tf.name)

    # remove lookup entry to avoid memory leak
    del _pairs_lookup[pairs_id]

    return valids
