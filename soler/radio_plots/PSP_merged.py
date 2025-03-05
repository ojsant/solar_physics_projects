import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors

from sunpy.time import TimeRange

from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

import cdflib
import soler.tools.cdaweb as cdaweb



def read_psp_fields_cdf(startdate, enddate, path=None):
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

    PSP_LFR_VARS = {'epoch':'epoch_lfr_stokes', 'frequency':'frequency_lfr_stokes', 'psd':'psp_fld_l3_rfs_lfr_PSD_SFU'}
    PSP_HFR_VARS = {'epoch':'epoch_hfr_stokes', 'frequency':'frequency_hfr_stokes', 'psd':'psp_fld_l3_rfs_hfr_PSD_SFU'}

    lfr_data = []
    hfr_data = []
    
    datasets = ['PSP_FLD_L3_RFS_LFR', 'PSP_FLD_L3_RFS_HFR']

    for dataset in datasets:
        data = []
        if dataset == 'PSP_FLD_L3_RFS_LFR':
            data = lfr_data
            variables = PSP_LFR_VARS
        else:
            data = hfr_data
            variables = PSP_HFR_VARS

        files = cdaweb.cdaweb_download_fido(dataset=dataset, startdate=startdate, enddate=enddate, path=path)

        freq_mhz  = cdflib.CDF(files[0]).varget(variables['frequency']) / 1e6 

        # PSP frequency array shape is (nTimeLFR, nFreqLFR)
        if freq_mhz.ndim == 2:
            freq_mhz = freq_mhz[0, :]

        psd_sfu = np.empty(shape=(0,len(freq_mhz)))
        time = np.array([])

        for file in files:
            cdf_file = cdflib.CDF(file)
            time_ns_1day  = cdf_file.varget(variables['epoch'])         # shape: (nTimeLFR,)
            psd_sfu_1day  = cdf_file.varget(variables['psd'])           # shape: (nTimeLFR, nFreqLFR)

            # remove bar artifacts caused by non-NaN values before time jumps
            # for each time step except the last one:
            for i in range(len(time_ns_1day)-1):
                # check if time increases by more than 60s:
                if time_ns_1day[i+1]-time_ns_1day[i] > 60000000000:
                    psd_sfu_1day[i,:] = np.nan

            time_dt  = cdflib.epochs.CDFepoch.to_datetime(time_ns_1day)
            time_mpl = mdates.date2num(time_dt)

            time = np.append(time, time_mpl)
            psd_sfu = np.append(psd_sfu, psd_sfu_1day, axis=0)

        data.append(time)
        data.append(freq_mhz)
        data.append(psd_sfu)

    df_lfr_psd = pd.DataFrame(lfr_data[2], index=lfr_data[0], columns=lfr_data[1])
    df_hfr_psd = pd.DataFrame(hfr_data[2], index=hfr_data[0], columns=hfr_data[1])

    return [df_lfr_psd, df_hfr_psd]


def plot_psp_fields(data, t_lims=None, cmap='jet'):
    """
    Plot PSP FIELDS data (both LFR and HFR). 
    """
    lfr_data = data[0]
    hfr_data = data[1]
    ###############################################################################
    # Build meshes for pcolormesh
    ###############################################################################
    TimeLFR2D, FreqLFR2D = np.meshgrid(lfr_data.index, lfr_data.columns, indexing='ij')
    TimeHFR2D, FreqHFR2D = np.meshgrid(hfr_data.index, hfr_data.columns, indexing='ij')

    # Log scale range
    vmin, vmax = 500, 1e7
    log_norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(10, 6),
        dpi=150,
        sharex=True
    )

    mesh_hfr = ax.pcolormesh(
        TimeHFR2D,
        FreqHFR2D,
        hfr_data,
        shading='nearest',
        cmap=cmap,
        norm=log_norm
    )
    ax.set_yscale('log')
    ax.set_ylabel("Frequency (MHz)", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)

    mesh_lfr = ax.pcolormesh(
        TimeLFR2D,
        FreqLFR2D,
        lfr_data,
        shading='nearest',
        cmap=cmap,
        norm=log_norm
    )

    
    ###############################################################################
    # Single colorbar on right
    ###############################################################################
    cbar = fig.colorbar(mesh_lfr, ax=ax,
                        orientation="vertical",
                        pad=0.02,
                        extend='both', extendfrac='auto')
    cbar.set_label("Intensity (sfu)", rotation=270, labelpad=10, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ###############################################################################
    # Use AutoDateLocator + ConciseDateFormatter, but no rotation on ticks
    ###############################################################################
    locator = AutoDateLocator()
    formatter = ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # We do NOT rotate the ticklabels (no overlap is guaranteed only if
    # your time range is not too large)
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=0, ha='center')

    # Set the x-range
    if t_lims is None:
        ax.set_xlim(lfr_data[0].index, lfr_data[-1].index)
    else:
        tr = TimeRange(t_lims)
        start_mpl = mdates.date2num(tr.start.to_datetime())
        end_mpl = mdates.date2num(tr.end.to_datetime())
        ax.set_xlim(start_mpl, end_mpl)
        
    plt.show()


if __name__ == "__main__":

    startdate = "2023/04/17 12:00"
    enddate   = "2023/04/18 16:00"

    data = read_psp_fields_cdf(startdate, enddate)

    plot_psp_fields(data, t_lims=(startdate, enddate))

