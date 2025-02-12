import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors

from sunpy.time import TimeRange

from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

import cdflib
import cdaweb

PSP_LFR_VARS = {'epoch':'epoch_lfr_stokes', 'frequency':'frequency_lfr_stokes', 'psd':'psp_fld_l3_rfs_lfr_PSD_SFU'}
PSP_HFR_VARS = {'epoch':'epoch_hfr_stokes', 'frequency':'frequency_hfr_stokes', 'psd':'psp_fld_l3_rfs_hfr_PSD_SFU'}


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

    # TODO If no end date is given, assume 1 day

    lfr_data = []
    hfr_data = []

    datasets = ['PSP_FLD_L3_RFS_LFR', 'PSP_FLD_L3_RFS_HFR']

    for dataset in datasets:
        data = []
        if dataset == 'PSP_FLD_L3_RFS_LFR':
            data = lfr_data
            var = PSP_LFR_VARS
        else:
            data = hfr_data
            var = PSP_HFR_VARS

        files = cdaweb.cdaweb_download_fido(dataset=dataset, startdate=startdate, enddate=enddate, path=path)

        freq  = cdflib.CDF(files[0]).varget(var['frequency'])     

        # PSP frequency array shape is (nTimeLFR, nFreqLFR)
        if freq.ndim == 2:
            freq = freq[0, :]

        psd_sfu = np.empty(shape=(0,len(freq)))
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

        psd_sfu = psd_sfu[:-1,:-1]     # remove last row and column

        data.append(time)
        data.append(freq)
        data.append(psd_sfu)

    return lfr_data, hfr_data


def plot_psp_fields(lfr_data, hfr_data, cmap='jet'):
    """
    Plot PSP FIELDS data (both LFR and HFR). 
    """

    time_hfr_mpl, freq_hfr_mhz, psd_hfr_sfu = hfr_data
    time_lfr_mpl, freq_lfr_mhz, psd_lfr_sfu = lfr_data

    ###############################################################################
    # Build meshes for pcolormesh
    ###############################################################################
    TimeLFR2D, FreqLFR2D = np.meshgrid(time_lfr_mpl, freq_lfr_mhz, indexing='ij')
    TimeHFR2D, FreqHFR2D = np.meshgrid(time_hfr_mpl, freq_hfr_mhz, indexing='ij')

    ###############################################################################
    # Custom colormap: gray for data < vmin, then Spectral/jet/whatever
    ###############################################################################
    # cmap = plt.get_cmap(cmap, 256)   
    # colors_combined = np.vstack((
    #     [0.5, 0.5, 0.5, 1.0], 
    #     cmap(np.linspace(0, 1, 256))
    # ))
    # custom_cmap = ListedColormap(colors_combined)

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
        psd_hfr_sfu,
        shading='flat',
        cmap=cmap,
        norm=log_norm
    )
    ax_hfr.set_yscale('log')
    ax_hfr.set_ylabel("HFR (MHz)", fontsize=8)
    ax_hfr.tick_params(axis='both', which='major', labelsize=8)

    
    ax_hfr.set_title("Parker Solar Probe FIELDS/RFS", fontsize=9)

    ###############################################################################
    # Plot LFR on bottom
    ###############################################################################
    mesh_lfr = ax_lfr.pcolormesh(
        TimeLFR2D,
        FreqLFR2D,
        psd_lfr_sfu,
        shading='flat',
        cmap=cmap,
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
    ax_lfr.set_xlim(time_lfr_mpl[0], time_lfr_mpl[-1])

    plt.show()


if __name__ == "__main__":

    startdate = "2023/04/17"
    enddate   = "2023/04/19"

    lfr_data, hfr_data = read_psp_fields_cdf(startdate, enddate)

    plot_psp_fields(lfr_data, hfr_data)

