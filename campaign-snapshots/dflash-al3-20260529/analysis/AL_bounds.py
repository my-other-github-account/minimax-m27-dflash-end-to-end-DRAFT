#!/usr/bin/env python3
"""Compute teacher-forced AL bounds from a DFlash drafter's per-position accept accuracies.

AL_LO (strict chain): 1 + sum of prefix-products of per-position accept probs.
AL_HI (independent ceiling): 1 + sum of per-position accept probs.

A *real* verified speculative-decode runtime AL must lie within [AL_LO, AL_HI].
Runtime AL ABOVE AL_HI is a red flag (looser runtime accept rule, OOD-easy prompts,
or a wrong AL denominator such as n_predict/(n_drafted/draft_max)).
"""

def al_bounds(p):
    prefix, al_lo = 1.0, 1.0
    for pi in p:
        prefix *= pi
        al_lo += prefix
    al_hi = 1.0 + sum(p)
    return al_lo, al_hi

if __name__ == "__main__":
    # step500 (093736 run, OUR from-scratch adapter), train per-position accept @ step500
    p_step500 = [0.302, 0.223, 0.163, 0.145, 0.131, 0.118, 0.122]
    lo, hi = al_bounds(p_step500)
    print(f"step500 per-position: {p_step500}")
    print(f"  AL_LO = {lo:.4f}   AL_HI = {hi:.4f}")
    print(f"  measured runtime: 2.43 (5p dmax13) .. 2.92 (3p dmax15)  -> ABOVE AL_HI {hi:.2f} = RED FLAG")

    # step600 val per-position (from val_in_epoch step_00000600.json)
    p_step600 = [0.3021, 0.2086, 0.1515, 0.1281, 0.1128, 0.1007, 0.0939]
    lo6, hi6 = al_bounds(p_step600)
    print(f"\nstep600 val per-position: {p_step600}")
    print(f"  AL_LO = {lo6:.4f}   AL_HI = {hi6:.4f}")
