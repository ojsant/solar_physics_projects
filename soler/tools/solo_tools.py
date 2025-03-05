import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os

import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib import cm

from seppy.loader.solo import mag_load
from seppy.tools import resample_df
from stixdcpy.quicklook import LightCurves
from sunpy.coordinates import get_horizons_coord
from sunpy.coordinates import frames

from tools.my_func_py3 import mag_angles, polarity_rtn
from tools.polarity_plotting import polarity_rtn, polarity_panel, polarity_colorwheel
from solo_epd_loader import epd_load, calc_ept_corrected_e, combine_channels
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# import warnings
# warnings.filterwarnings('ignore')

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('axes', titlesize=20)  # fontsize of the axes title
plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
plt.rcParams['agg.path.chunksize'] = 20000

import sunpy.net.attrs as a
import sunpy_soar  
from sunpy.net import Fido
from sunpy.timeseries import TimeSeries

def swa_load_grnd_mom(startdate, enddate, path=None):
    """
    Load SolO/SWA L2 ground moments

    Load-in data for Solar Orbiter/SWA sensor ground moments. Supports level 2
    provided by ESA's Solar Orbiter Archive. Optionally downloads missing
    data directly. Returns data as Pandas dataframe.

    Parameters
    ----------
    startdate, enddate : {datetime, str, or int}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)),
        "standard" datetime string (e.g., "2021/04/15") or integer of the form
        yyyymmdd with empty positions filled with zeros, e.g. '20210415'
        (enddate must always be later than startdate)
    path : {str}, optional
        Local path for storing downloaded data, by default None

    Returns
    -------
    Pandas dataframe
    """    
    instrument = a.Instrument('SWA')
    time = a.Time(startdate, enddate)
    level = a.Level(2)
    product = a.soar.Product('SWA-PAS-GRND-MOM')
    
    result = Fido.search(instrument & time & level & product)
    files = Fido.fetch(result,path=path)
    
    solo_swa = TimeSeries(files, concatenate=True)
    df_solo_swa = solo_swa.to_dataframe()
    return df_solo_swa


def rpw_load(dataset, startdate, enddate, path=None):
    """
    rpw-tnr-surv or rpw-hfr-surv
    """
    instrument = a.Instrument('RPW')
    time = a.Time(startdate, enddate)
    level = a.Level(2)
    product = a.soar.Product(dataset)

    result = Fido.search(instrument & time & level & product)
    files = Fido.fetch(result,path=path)
    solo_rpw = TimeSeries(files, concatenate=True)
    df_solo_rpw = solo_rpw.to_dataframe()
    return df_solo_rpw