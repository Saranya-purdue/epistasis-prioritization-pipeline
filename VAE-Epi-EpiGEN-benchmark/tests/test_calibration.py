import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'01_calibrate_epigen_alpha.py'
spec=importlib.util.spec_from_file_location('cal',SCRIPT)
cal=importlib.util.module_from_spec(spec); spec.loader.exec_module(cal)

def test_minor_allele_dosage_flips_high_frequency():
    g=pd.Series([2,2,1,2])
    x=cal.minor_allele_dosage(g)
    assert np.allclose(x,[0,0,1,0])

def test_joint_dominant_risk():
    G={'a':np.array([0.,1.,1.]),'b':np.array([2.,0.,1.])}
    pairs=pd.DataFrame({'SNP1':['a'],'SNP2':['b']})
    risk=cal.calculate_risk(G,pairs,'joint-dominant',2.0)
    assert np.allclose(risk,[1.,1.,2.])

def test_calibration_hits_target():
    rng=np.random.default_rng(1)
    G={'a':rng.binomial(2,.3,2000).astype(float),'b':rng.binomial(2,.35,2000).astype(float)}
    pairs=pd.DataFrame({'SNP1':['a'],'SNP2':['b']})
    _, frac, _=cal.find_alpha(G,pairs,'multiplicative',target=.10,noise_sd=1.0,tolerance=5e-4)
    assert abs(frac-.10)<1e-3
