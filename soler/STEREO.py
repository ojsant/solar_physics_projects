import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors

from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

import cdflib
import os
import sunpy
import cdaweb

def load_swaves(dataset, startdate, enddate, path=None):
    """
    Load PSP Fields data from CDAWeb. Combines time if timespan is multiple days.

    Parameters
    ----------
    startdate, enddate : datetime or str
        start/end date in standard format (e.g. YYYY-mm-dd or YYYY/mm/dd, anything parseable by sunpy.time.parse_time)
    
    dataset : string (optional)
        dataset identifier (PSP_FLD_L3_RFS_HFR for high frequency data and  for low) (both by default)
                        
    Returns
    -------
    ndarray :
        1) timestamps in matplotlib format,
        2) frequencies in MHz,
        3) intensities in sfu for each (time, frequency) data point
    """


    files = cdaweb.cdaweb_download_fido(dataset=dataset, startdate=startdate, enddate=enddate, path=path)

    freq_mhz = cdflib.CDF(files[0]).varget("FREQUENCY") / 1e6

    psd_sfu = np.empty(shape=(0,len(freq_mhz)))

    time = np.array([], dtype="datetime64")

    for file in files:
        cdf_file = cdflib.CDF(file)
        
        time_ns_1day = cdf_file.varget('Epoch')
        time_dt  = cdflib.epochs.CDFepoch.to_datetime(time_ns_1day)
        psd_sfu_1day  = cdf_file.varget('PSD_SFU')

        time = np.append(time, time_dt)
        psd_sfu = np.append(psd_sfu, psd_sfu_1day, axis=0)

    # remove bar artifacts caused by non-NaN values before time jumps
        # for each time step except the last one:
    for i in range(len(time)-1):
        # check if time increases by more than 5 min:
        if time[i+1] - time[i] > np.timedelta64(5, "m"):
            psd_sfu[i,:] = np.nan

    psd_sfu = pd.DataFrame(psd_sfu, index=time, columns=freq_mhz)

    return psd_sfu