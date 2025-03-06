# TODO: 
# - radio plots

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

opt  = dict(ept_l3 = True,
            av_en = False,
            ion_conta_corr = False,

            data_path = f"{os.getcwd()}/data/",

            startdate = dt.datetime(2022, 10, 3),
            enddate = dt.datetime(2022, 10, 5),

            # date = f"{startdate.year}{startdate.month:02d}{startdate.day:02d}"
            viewing = 'sun',

            resample = '1min',
            resample_particles = '5min',

            stix_ltc = True,  # correct SolO/STIX data for light travel time

            pos_timestamp = None,  #'center'

            plot_radio = True,
            plot_stix = True,
            plot_electrons = False,
            plot_protons = True,
            plot_mag_angles = True, 
            plot_mag = True,
            plot_Vsw = True,
            plot_N = True,
            plot_T = True,
            plot_polarity = True
)



def load_data(opt):
    data = {}
    if plot_electrons or plot_protons:
        if ept_l3:
            df_ept_org, df_rtn_ept, df_hci_ept, energies_ept, metadata_ept = epd_load(sensor='ept', level='l3', pos_timestamp=pos_timestamp,
                                                                                    startdate=startdate, enddate=enddate,
                                                                                    autodownload=True, path=data_path)
        else:
            protons_ept, electrons_ept, energies_ept = epd_load(sensor='ept', level='l2', startdate=startdate, enddate=enddate, 
                                                                pos_timestamp=pos_timestamp,viewing=viewing, path=data_path, autodownload=True)
        protons_het, electrons_het, energies_het = epd_load(sensor='het', level='l2', startdate=startdate, enddate=enddate, 
                                                            pos_timestamp=pos_timestamp,viewing=viewing, path=data_path, autodownload=True)
    if plot_stix:
        lc = LightCurves.from_sdc(start_utc=startdate, end_utc=enddate, ltc=stix_ltc)
        df_stix_orig = lc.to_pandas()    

    if plot_mag or plot_mag_angles or plot_polarity:
        mag_data_org = mag_load(startdate, enddate, level='l2', frame='rtn', path=data_path)
        mag_data_org['Bmag'] = np.sqrt(np.nansum((mag_data_org.B_RTN_0.values**2, mag_data_org.B_RTN_1.values**2, mag_data_org.B_RTN_2.values**2), axis=0))

    if plot_Vsw or plot_N or plot_T:
        swa_data = swa_load_grnd_mom(startdate, enddate, path=data_path)
        swa_vsw = np.sqrt(swa_data.V_RTN_0**2 + swa_data.V_RTN_1**2 + swa_data.V_RTN_2**2)
        swa_data['vsw'] = swa_vsw

        temp = np.sqrt(swa_data.TxTyTz_RTN_0**2 + swa_data.TxTyTz_RTN_2**2 + swa_data.TxTyTz_RTN_2**2)
        swa_data['temp'] = temp

    # if plot_radio:
    #     df_hfr = rpw_load("RPW-HFR-SURV".lower(), startdate=startdate, enddate=enddate, path=data_path)
    #     df_tnr = rpw_load("RPW-TNR-SURV".lower(), startdate=startdate, enddate=enddate, path=data_path)

    if plot_electrons or plot_protons:
        df_electrons_het = resample_df(electrons_het, resample_particles, pos_timestamp=pos_timestamp)
        df_protons_het = resample_df(protons_het, resample_particles, pos_timestamp=pos_timestamp)

        if ept_l3:
            df_ept = resample_df(df_ept_org, resample_particles, pos_timestamp=pos_timestamp)
            if viewing.lower() == 'south':
                view = 'D'
            else:
                view = viewing[0].upper()
        else:
            df_electrons_ept = resample_df(electrons_ept, resample_particles, pos_timestamp=pos_timestamp)
            df_protons_ept = resample_df(protons_ept, resample_particles, pos_timestamp=pos_timestamp)

    if plot_Vsw or plot_N or plot_T:
        df_swa = resample_df(swa_data, resample, pos_timestamp=pos_timestamp)

    if plot_mag:
        mag_data = resample_df(mag_data_org, resample, pos_timestamp=pos_timestamp)

    if plot_stix:
        df_stix = resample_df(df_stix_orig, resample, pos_timestamp=pos_timestamp) 

    # correct EPT level 2 electron data for ion contamination:
    if plot_electrons and ion_conta_corr and not ept_l3:
        # df_electrons_ept2 = calc_EPT_corrected_e(df_electrons_ept['Electron_Flux'], df_protons_ept['Ion_Flux'])
        df_electrons_ept = calc_ept_corrected_e(df_electrons_ept, df_protons_ept)
        df_electrons_ept = df_electrons_ept.mask(df_electrons_ept < 0)

    print('EPT electron channels:')
    for i, e in enumerate(energies_ept['Electron_Bins_Text']):
        print(i, e)
    print('')
    print('HET electron channels:')
    for i, e in enumerate(energies_het['Electron_Bins_Text']):
        print(i, e)
    print('')
    print('EPT ion channels:')
    for i, e in enumerate(energies_ept['Ion_Bins_Text']):
        print(i, e)
    print('')
    print('HET ion channels:')
    for i, e in enumerate(energies_het['H_Bins_Text']):
        print(i, e)

    ept_ele_channels = [0, 3, 6, 9, 12, 15]
    het_ele_channels = [0, 1, 2, 3]
    ept_ion_channels = [0, 5, 10, 15, 20, 25, 30]
    het_ion_channels = [0, 5, 10, 15, 20, 25, 30, 35]
    return data

