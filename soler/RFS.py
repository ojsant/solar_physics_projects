import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

import cdflib
import datetime as dt
import cdaweb

PSP_LFR_VARS = {'epoch':'epoch_lfr_stokes', 'frequency':'frequency_lfr_stokes', 'psd':'psp_fld_l3_rfs_lfr_PSD_SFU'}
PSP_HFR_VARS = {'epoch':'epoch_hfr_stokes', 'frequency':'frequency_hfr_stokes', 'psd':'psp_fld_l3_rfs_hfr_PSD_SFU'}
#STA_VARS = {'epoch' : 'Epoch', 'frequency': 'FREQUENCY', 'psd':'PSD_SFU'}


###############################################################################
# Function: Convert J2000 ns -> Python datetime
###############################################################################
# def j2000_ns_to_datetime(ns_array):
#     """
#     Convert 'nanoseconds since 2000-01-01T12:00:00' (J2000)
#     into a numpy array of Python datetime objects.
#     """
#     j2000_ref = datetime.datetime(2000, 1, 1, 12, 0, 0)
#     return np.array([
#         j2000_ref + datetime.timedelta(seconds=(ns * 1e-9))
#         for ns in ns_array
#     ])




#def fillvals_to_nan():


def read_psp_fields_files(dataset, startdate, enddate):
    """
    Load PSP Fields data from CDAWeb. Combines time if timespan is multiple days.

    Parameters
    ----------
        dataset : string
            dataset identifier (PSP_FLD_L3_RFS_HFR for high frequency data and PSP_FLD_L3_RFS_LFR for low)
                            
        startdate, enddate : datetime or str
            start/end date
    
    Returns:
        ndarray
          timestamps in matplotlib format
        (ndarray): frequencies in MHz
        (ndarray): intensities in sfu for each (time, frequency) data point
    """
    # use instrument specific variable names
    dset_split = dataset.split("_")
    
    if 'LFR' in dset_split:
        var = PSP_LFR_VARS
    elif 'HFR' in dset_split:
        var = PSP_HFR_VARS
    

    files = cdaweb.cdaweb_download_fido(dataset=dataset, startdate=startdate, enddate=enddate)

    freq  = cdflib.CDF(files[0]).varget(var['frequency'])     

    # PSP frequency array shape is (nTimeLFR, nFreqLFR)
    if var == PSP_HFR_VARS or var == PSP_LFR_VARS:
        freq = freq[0, :]  
    
    psd_sfu = np.zeros((1, len(freq)))  # this is just to get the right shape for appending, could probably be done better
    time = np.array([])


    for file in files:
        cdf_file = cdflib.CDF(file)
        time_ns_1day  = cdf_file.varget(var['epoch'])         # shape: (nTimeLFR,)
        psd_sfu_1day  = cdf_file.varget(var['psd'])           # shape: (nTimeLFR, nFreqLFR)

        # cdf_lfr.close()  # I think this is outdated
        # Remove bar-like artifacts from PSP high frequency data (pcolormesh extends non-NaN values over timejumps)
        if var == PSP_HFR_VARS:
            # for each time step except the last one:
            for i in range(len(time_ns_1day)-1):
                # check if time increases by more than 60s:
                if time_ns_1day[i+1]-time_ns_1day[i] > 60000000000:
                    psd_sfu_1day[i,:] = np.nan

        # time_lfr_dt  = j2000_ns_to_datetime(time_lfr_ns)
        time_dt  = cdflib.epochs.CDFepoch.to_datetime(time_ns_1day)
        time_mpl = mdates.date2num(time_dt)

        time = np.append(time, time_mpl)
        psd_sfu = np.append(psd_sfu, psd_sfu_1day, axis=0)

    psd_sfu = psd_sfu[1:-1,:-1]     # remove the zero row + last row and column

    return time, freq, psd_sfu


def plot_data(lfr_data, hfr_data, cmap, sc="PSP"):
    
    ###############################################################################
    # Build meshes for pcolormesh
    ###############################################################################
    TimeLFR2D, FreqLFR2D = np.meshgrid(lfr_data[0], lfr_data[1], indexing='ij')
    TimeHFR2D, FreqHFR2D = np.meshgrid(hfr_data[0], hfr_data[1], indexing='ij')
    psdLFR = lfr_data[2]
    psdHFR = hfr_data[2]

    ###############################################################################
    # Custom colormap: gray for data < vmin, then Spectral/jet/whatever
    ###############################################################################
    cmap = plt.get_cmap(cmap, 256)   
    colors_combined = np.vstack((
        [0.5, 0.5, 0.5, 1.0], 
        cmap(np.linspace(0, 1, 256))
    ))
    custom_cmap = ListedColormap(colors_combined)

    # Log scale range
    vmin, vmax = 500, 1e7
    log_norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    ###############################################################################
    # Create figure: 2 subplots, no vertical space between them
    ###############################################################################
    fig, (ax_hfr, ax_lfr) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(10, 6),
        dpi=150,
        sharex=True
    )

    # Adjust subplot params to remove vertical space
    fig.subplots_adjust(
        left=0.1, right=0.98,
        top=0.92, bottom=0.12,
        hspace=0.0  # <= no space between subplots
    )

    ###############################################################################
    # Plot HFR on top
    ###############################################################################
    mesh_hfr = ax_hfr.pcolormesh(
        TimeHFR2D,
        FreqHFR2D,
        psdHFR,
        shading='flat',
        cmap=custom_cmap,
        norm=log_norm
    )
    ax_hfr.set_yscale('log')
    ax_hfr.set_ylabel("HFR (MHz)", fontsize=8)
    ax_hfr.tick_params(axis='both', which='major', labelsize=8)

    # TODO: instrument specific name
    if sc == 'PSP':
        ax_hfr.set_title("Parker Solar Probe FIELDS/RFS", fontsize=9)

    ###############################################################################
    # Plot LFR on bottom
    ###############################################################################
    mesh_lfr = ax_lfr.pcolormesh(
        TimeLFR2D,
        FreqLFR2D,
        psdLFR,
        shading='flat',
        cmap=custom_cmap,
        norm=log_norm
    )
    ax_lfr.set_yscale('log')
    ax_lfr.set_ylabel("LFR (MHz)", fontsize=8)
    ax_lfr.set_xlabel("Time (UTC)", fontsize=8)
    ax_lfr.tick_params(axis='both', which='major', labelsize=8)

    
    ###############################################################################
    # Single colorbar on right
    ###############################################################################
    cbar = fig.colorbar(mesh_lfr, ax=[ax_hfr, ax_lfr],
                        orientation="vertical",
                        pad=0.02,
                        extend='both', extendfrac='auto')
    cbar.set_label("Intensity (sfu)", rotation=270, labelpad=10, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    #cbar.cmap.set_under('gray')

    ###############################################################################
    # Use AutoDateLocator + ConciseDateFormatter, but no rotation on ticks
    ###############################################################################
    locator = AutoDateLocator()
    formatter = ConciseDateFormatter(locator)
    ax_lfr.xaxis.set_major_locator(locator)
    ax_lfr.xaxis.set_major_formatter(formatter)

    # We do NOT rotate the ticklabels (no overlap is guaranteed only if
    # your time range is not too large)
    for label in ax_lfr.get_xticklabels(which='major'):
        label.set(rotation=0, ha='center')

    # Set the x-range
    ax_lfr.set_xlim(lfr_data[0][0], lfr_data[0][-1])

    plt.show()


if __name__ == "__main__":
    lfr_data = read_psp_fields_files("PSP_FLD_L3_RFS_LFR", startdate="2019/04/17", enddate="2019/04/19")
    hfr_data = read_psp_fields_files("PSP_FLD_L3_RFS_HFR", startdate="2019/04/17", enddate="2019/04/19")

    plot_data(lfr_data, hfr_data, cmap='jet')

