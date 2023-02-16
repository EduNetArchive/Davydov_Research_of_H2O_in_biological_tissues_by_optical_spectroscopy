import numpy as np
from collections import namedtuple
from scipy.interpolate import interp1d
import scipy.io as scio
import pandas as pd


"""
Data on optical properties of tissues were taken from
"Jacques, S. L. (2013). Optical properties of biological tissues: a review. 
Physics in Medicine & Biology, 58(11), R37."


muafat was taken from Veen et al. 2004
Van Veen, R. L. P., et al. "Determination of visible near-IR absorption coefficients of mammalian fat using time-and spatially resolved diffuse reflectance and transmission spectroscopy." Journal of biomedical optics 10.5 (2005): 054004.

"""

TissueProperties = namedtuple('TissueProperties','name mua mus g')

def _read_od_spectra(path):
    return pd.read_csv(path,index_col=0)

_od_paths = ['temp/temp/water_OD.csv','temp/temp/hbo2_OD.csv','temp/temp/hb_OD.csv','temp/temp/melanin_OD.csv','temp/temp/muafat.csv']
_names = ['muawater','muaoxy','muadeoxy','muamel','muafat']

_spectra = {}
for component_name, path in zip(_names,_od_paths): 
    _spectra[component_name] = _read_od_spectra(path)

_interpolated_spectra = {k: interp1d(_spectra[k].index.values,_spectra[k].iloc[:,0].values) for k in _spectra}

def _get_standard_tissue(wavelength_nm,B,S,W,M,F,musp500,fray,bmie,gg):
    MUA = np.array([_interpolated_spectra[k](wavelength_nm) for k in ['muaoxy','muadeoxy','muawater','muamel','muafat']])
    musp = musp500*(fray*(wavelength_nm/500)**-4 + (1-fray)*(wavelength_nm/500)**-bmie)
    X = np.array([B*S, B*(1-S), W, M, F])
    mua = np.sum(MUA*X)
    mus = musp/(1-gg)
    g   = gg

    return mua, mus, g
    
    
def tissue_id_to_name():
    return {0:'air',1:'water',2:'blood',3:'dermis',4:'epidermis',5:'subcutaneous-fat'}

def make_tissue_list(wavelength_nm):
    """
    generate list of tissue optical properties for given wavelength_nm
    350 <wavelength_nm < 1000 nm
    
    Implementation of makeTissueList.m in python
    
    return [air,water,blood,dermis,epidermis]
    """
    
    MUA = np.array([_interpolated_spectra[k](wavelength_nm) for k in ['muaoxy','muadeoxy','muawater','muamel']])
    air = TissueProperties(name='air',mua=0.0001,mus=1.0,g=1.0)
    water = TissueProperties(name='water',mua=MUA[2],mus=10,g=1)
    
    
    #Vein blood according to OMLC mcxyz example
    mua,mus,g = _get_standard_tissue(wavelength_nm,B=1.0,S=0.75,W=0.95,M=0.,F=0.,musp500=22.,fray=0.0,bmie=1.0,gg=.9)
    blood = TissueProperties(name='blood',mua=mua,mus=mus,g=g)
    
    # Dermis scattering as according to Salomatina et al. 2006 and Jacques 2013 review
    mua,mus,g = _get_standard_tissue(wavelength_nm,B=0.002,S=0.67,W=0.65,M=0.,F=0.00,musp500=48.0,fray=0.41,bmie=0.562,gg=0.9)
    dermis = TissueProperties(name='dermis',mua=mua,mus=mus,g=g)
    
    #Epiderims according to Salomatina et al. and Jacques 2013 review
    mua,mus,g = _get_standard_tissue(wavelength_nm,B=0,S=0.5,W=0.5,M=0.03,F=0.,musp500=66.7,fray=0.29,bmie=0.689,gg=0.9)
    epidermis = TissueProperties(name='epidermis',mua=mua,mus=mus,g=g)
    
    #Adipose tissue with \ 
    mua,mus,g =  _get_standard_tissue(wavelength_nm,B=0.,S=0.5,W=0.4,M=0.,F=.6,
                                      musp500=15.4,fray=0.,bmie=0.68,gg=0.9)
    subcutaneous_adipose = TissueProperties(name='subcutaneous-adipose',mua=mua,mus=mus,g=g)
    
    return [air,water,blood,dermis,epidermis,subcutaneous_adipose]

    