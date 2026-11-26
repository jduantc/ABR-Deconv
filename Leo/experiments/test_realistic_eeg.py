"""Tests for the processed forward operator and known-truth data identity."""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import sosfilt

sys.path.insert(0,str(Path(__file__).resolve().parent))
import realistic_eeg as r


def test_forward_operator_matches_filtering_convolved_signal():
    src=r.source('sub-exp301')
    matrices,p,regular,meta=r.pulse_source(src)
    ctx=np.zeros(r.GUARD+r.N,np.float32);ctx[r.GUARD:]=regular
    x=r.stimulus_matrix(ctx).numpy().astype(np.float64)
    h=np.random.default_rng(27).normal(size=256)
    expected=sosfilt(r.filters(),x@h)[r.GUARD:]
    expected-=expected.mean()
    np.testing.assert_allclose(matrices['regular']@h,expected,atol=1e-10,rtol=1e-9)


def test_quiet_sources_and_simulation_truth_are_consistent():
    d,meta,_=r.build('sub-exp302',2,97501,815001)
    assert meta['real_pulse_count']!=meta['regular_pulse_count']
    for w in meta['quiet_windows']:
        assert w['end_sample']<=meta['first_playback_sample']-5000
    for obs in d.values():
        np.testing.assert_allclose(obs['eeg']-obs['noise'],obs['truth']@obs['matrix'].T,atol=1e-12)
        assert np.all(obs['truth'][-1]==0)
    np.testing.assert_allclose(r.rms(d['real_matched_white']['noise'],axis=1),r.rms(d['real_quiet_native']['noise'],axis=1),rtol=1e-10)


def test_spectral_weighting_preserves_linear_forward_residual():
    from realistic_eeg_whitening import transform
    rng=np.random.default_rng(56)
    matrix=rng.normal(size=(r.N,256))
    h=rng.normal(size=256)
    y=rng.normal(size=(2,r.N))
    weights=np.linspace(.2,2.,r.N//2+1)
    transformed=transform({'matrix':matrix,'eeg':y},weights)
    expected=np.fft.irfft(np.fft.rfft(y-matrix@h,axis=1)*weights[None,:],n=r.N,axis=1)
    np.testing.assert_allclose(transformed['eeg']-transformed['matrix']@h,expected,atol=1e-10,rtol=1e-9)
