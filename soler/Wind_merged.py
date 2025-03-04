import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors

from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

import cdflib
import os
import sunpy

from sunpy.net import Scraper
from sunpy.time import TimeRange
from sunpy.data.data_manager.downloader import ParfiveDownloader

def download_wind_waves_cdf(sensor, startdate, enddate, path=None):
    """
    Download Wind WAVES L2 data files for given time range.

    Parameters
    ----------
    sensor: str
        RAD1 or RAD2 (lower case works as well)
    startdate, enddate: str or dt
        start and end dates as parse_time compatible strings or datetimes (see TimeRange docs)
    path : str (optional)
        Local download directory, defaults to sunpy's data directory
    
    Returns
    -------
    List of downloaded files
    """
    dl = ParfiveDownloader()
    
    timerange = TimeRange(startdate, enddate)

    try:
        pattern = "https://spdf.gsfc.nasa.gov/pub/data/wind/waves/{sensor}_l2/%Y/wi_l2_wav_{sensor}_%Y%m%d_{version}.cdf"

        scrap = Scraper(pattern=pattern, sensor=sensor.lower(), version="v\\d{2}")  # regex matching "v{any digit}{any digit}""
        filelist_urls = scrap.filelist(timerange=timerange)

        filelist_urls.sort()

        # After sorting, any multiple versions are next to each other in ascending order.
        # If there are files with same dates, assume multiple versions -> pop the first one and repeat.
        # Should end up with a list with highest version numbers. Magic number -7 is the index where 
        # version number starts
        # As of 13.2.2025, no higher versions than v01 exist in either rad1_l2 or rad2_l2 directory

        i = 0
        while i < len(filelist_urls) - 1:
            if filelist_urls[i+1][:-7] == filelist_urls[i][:-7]:
                filelist_urls.pop(i)
            else:
                i += 1

        filelist = [url.split('/')[-1] for url in filelist_urls]

        if path is None:
            filelist = [sunpy.config.get('downloads', 'download_dir') + os.sep + file for file in filelist]
        elif type(path) is str:
            filelist = [path + os.sep + f for f in filelist]
        downloaded_files = filelist

        # Check if file with same name already exists in path
        for url, f in zip(filelist_urls, filelist):
            if os.path.exists(f) and os.path.getsize(f) == 0:
                os.remove(f)
            if not os.path.exists(f):
                dl.download(url=url, path=f)

    except (RuntimeError, IndexError):
        print(f'Unable to obtain Wind WAVES {sensor} data for {startdate}-{enddate}!')
        downloaded_files = []

    return downloaded_files


def load_waves_rad(dataset, startdate, enddate, file_path=None):
    """
    Read Wind/WAVES data (assuming freq is 1D or identical rows if 2D)

    Parameters
    ----------
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or
        "standard" datetime string (e.g., "2021/04/15") (enddate must always be
        later than startdate)
    file_path : {str}, optional
        File path as a string. Defaults to sunpy's default download directory

    """
    
    files = download_wind_waves_cdf(dataset, startdate, enddate, path=file_path)

    # Read the frequency binning (assumed constant across all data)
    freq_hz  = cdflib.CDF(files[0]).varget("FREQUENCY")

    # If freq is 2D but each row is identical, take freq_raw[0,:]
    if freq_hz.ndim == 2:
        freq_hz = freq_hz[0, :]
    
    psd_v2hz = np.empty(shape=(0,len(freq_hz))) 
    time_dt = np.array([], dtype="datetime64")

    # append data 
    for file in files:
        cdf = cdflib.CDF(file)

        # PSD shape (nTime, nFreq)
        psd_raw = cdf.varget("PSD_V2_SP")
        # Time
        time_ns = cdf.varget("Epoch")  # shape (nTime,)

        time_dt = np.append(time_dt, cdflib.epochs.CDFepoch.to_datetime(time_ns))

        psd_v2hz = np.append(psd_v2hz, psd_raw, axis=0)

    # remove bar artifacts caused by non-NaN values before time jumps
    # for each time step except the last one:
    for i in range(len(time_dt)-1):
        # check if time increases by more than 5 min:
        if time_dt[i+1] - time_dt[i] > np.timedelta64(5, "m"):
            psd_v2hz[i,:] = np.nan

    # Some files use a fill value ~ -9.9999998e+30
    fill_val = -9.999999848243207e+30
    valid_mask = (freq_hz > 0) & (freq_hz != fill_val) 
    freq_hz = freq_hz[valid_mask]
    psd_v2hz = psd_v2hz[:, valid_mask]

    # Convert frequency to MHz
    freq_mhz = freq_hz / 1e6

    # Sort time
    if not sorted(time_dt):
        idx_t = np.argsort(time_dt)
        time_dt = time_dt[idx_t]
        psd_v2hz  = psd_v2hz[idx_t, :]

    # Remove duplicate times
    t_unique, t_uidx = np.unique(time_dt, return_index=True)
    if len(t_unique) < len(time_dt):
        time_dt = t_unique
        psd_v2hz  = psd_v2hz[t_uidx, :]

    # Sort freq
    if not sorted(freq_mhz):
        idx_f = np.argsort(freq_mhz)
        freq_mhz = freq_mhz[idx_f]
        psd_v2hz  = psd_v2hz[:, idx_f]

    # Remove duplicate freqs
    f_unique, f_uidx = np.unique(freq_mhz, return_index=True)
    if len(f_unique) < len(freq_mhz):
        freq_mhz = f_unique
        psd_v2hz  = psd_v2hz[:, f_uidx]

    data = pd.DataFrame(psd_v2hz, index=time_dt, columns=freq_mhz)

    return data


def plot_wind_waves(rad1_data, rad2_data, t_lims=None, cmap='jet'):
    """
    Plot Wind WAVES data (both RAD1 and RAD2).
    """


    ###############################################################################
    # 1) Read both RAD2 (top) and RAD1 (bottom)
    ###############################################################################

    TT2, FF2 = np.meshgrid(rad2_data.index, rad2_data.columns, indexing='ij')
    TT1, FF1 = np.meshgrid(rad1_data.index, rad1_data.columns, indexing='ij')

    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(10,6),
        dpi=150,
        sharex=True  # share time axis
    )

    #Make the following auto-range if you are automating things
    vmin, vmax = 1e-15, 1e-9  # Adjust to your data range
    log_norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # note: 'flat' shading option has to be used when handling time jumps without pandas resampling
    # -> remove last rows and columns, hence the iloc[:-1,:-1]

    # --- RAD2 subplot (TOP) ---
    mesh2 = ax.pcolormesh(
        TT2, FF2,
        rad2_data.iloc[:-1, :-1],
        shading='flat',
        cmap=cmap,
        norm=log_norm
    )

    # --- RAD1 subplot (BOTTOM) ---
    mesh1 = ax.pcolormesh(
        TT1, FF1,
        rad1_data.iloc[:-1, :-1],
        shading='flat',
        cmap=cmap,
        norm=log_norm
    )

    ax.set_yscale('log')
    ax.set_ylabel("Frequency [MHz]", fontsize=8)
    ax.set_xlabel("Time (UTC)", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    # Shared colorbar for both subplots on the right
    cbar = fig.colorbar(mesh2, ax=ax, orientation="vertical", pad=0.02, extend='both')
    cbar.set_label("PSD (V^2/Hz)", rotation=270, labelpad=12, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Format time axis on the bottom subplot
    locator = AutoDateLocator()
    formatter = ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=0, ha='center')

    if t_lims is None:
        ax.set_xlim(rad1_data.index[0], rad1_data.index[-1])
    else:
        tr = TimeRange(t_lims)
        start_mpl = mdates.date2num(tr.start.to_datetime())
        end_mpl = mdates.date2num(tr.end.to_datetime())
        ax.set_xlim(start_mpl, end_mpl)

    plt.show()

if __name__ == "__main__":
    startdate = "2024/06/17"
    enddate   = "2024/06/18"

    rad1_data = load_waves_rad(dataset = "RAD1", startdate=startdate, enddate=enddate)
    rad2_data = load_waves_rad(dataset = "RAD2", startdate=startdate, enddate=enddate)

    plot_wind_waves(rad1_data, rad2_data)