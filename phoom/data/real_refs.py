"""Paper-method 35 min responses for each subject and story, used as the
training set for the real-data priors (real_prior.py).

    python real_refs.py sub-exp302 sub-exp304 ...

Writes refs/<sub>_<story>.npy, shape (10, NK), uV, after the 150 Hz highpass.
sub-exp303 is skipped: its second run repeats stimuli (211 triggers per
story, 210 stimulus files) so the trial table doesn't line up with the audio.
"""
import sys
from pathlib import Path

import numpy as np

import real_data as rd

OUT = Path(__file__).parent / "refs"
SKIP = {"sub-exp303"}


def main(subs):
    OUT.mkdir(exist_ok=True)
    for sub in subs:
        if sub in SKIP:
            print(f"{sub}: skipped, see docstring")
            continue
        y_all, starts, clock_adjust = rd.load_subject(sub)
        for story in ["wizard", "wonderland", "drdolittle"]:
            dst = OUT / f"{sub}_{story}.npy"
            if dst.exists():
                continue
            trials = starts[starts.stimulus_type == story].reset_index(drop=True)
            acc = rd.accumulate(y_all, trials, story, clock_adjust, [])
            ref = rd.post_hp(rd.paper_estimate(acc.snaps[acc.n]))
            np.save(dst, ref)
            amps = [rd.wave_v(r)[1] for r in ref]
            print(f"{sub} {story}: {acc.n} trials -> {dst.name}, wave V amps {np.round(amps, 3)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
