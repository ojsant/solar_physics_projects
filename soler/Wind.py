import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
#from scipy.interpolate import RegularGridInterpolator
import cdflib
import datetime as dt
import cdaweb

#def fillvals_to_nan(array):


# def custom_cmap(cmap='Spectral'):
#     cmap_spectral = plt.get_cmap(cmap, 256)
#     colors_combined = np.vstack((
#         [0.5, 0.5, 0.5, 1.0],  # color for <vmin
#         cmap_spectral(np.linspace(0, 1, 256))
#     ))
#     return ListedColormap(colors_combined)

###############################################################################
# 1) Convert J2000 ns -> Python datetime
###############################################################################
def j2000_ns_to_datetime(ns_array):
    """
    Convert 'nanoseconds since 2000-01-01T12:00:00' (J2000)
    into a numpy array of Python datetime objects.
    """
    j2000_ref = dt.datetime(2000, 1, 1, 12, 0, 0)
    return np.array([
        j2000_ref + dt.timedelta(seconds=(ns * 1e-9))
        for ns in ns_array
    ])


###############################################################################
# 2) Read a single Wind/WAVES file (assuming freq is 1D or identical rows if 2D)
###############################################################################
def read_wind_waves_cdf(dataset, startdate, enddate, file_path=None, psd_var="PSD_V2_SP"):
    files = cdaweb.download_wind_waves_cdf(dataset, startdate, enddate, path=file_path)
    
    # Read the frequency binning (assumed constant across all data)
    freq_hz  = cdflib.CDF(files[0]).varget("FREQUENCY")

    # If freq is 2D but each row is identical, take freq_raw[0,:]
    if freq_hz.ndim == 2:
        freq_hz = freq_hz[0, :]
    
    psd_v2hz = np.empty(shape=(0,len(freq_hz))) 
    time = np.array([])

    # append data 
    for file in files:
        cdf = cdflib.CDF(file)
        
        # Time
        time_ns = cdf.varget("Epoch")  # shape (nTime,)
        time_dt = j2000_ns_to_datetime(time_ns)
        time_mpl = mdates.date2num(time_dt)
        
        # PSD shape (nTime, nFreq)
        psd_raw = cdf.varget(psd_var)
        #cdf.close()
    
        time = np.append(time, time_mpl)
        psd_v2hz = np.append(psd_v2hz, psd_raw, axis=0)

    # Some files use a fill value ~ -9.9999998e+30
    fill_val = -9.999999848243207e+30
    valid_mask = (freq_hz > 0) & (freq_hz != fill_val) 
    freq_hz = freq_hz[valid_mask]
    psd_v2hz = psd_v2hz[:, valid_mask]

    # Convert frequency to MHz
    freq_mhz = freq_hz / 1e6

    # Sort time
    if not sorted(time):
        idx_t = np.argsort(time)
        time = time[idx_t]
        psd_v2hz  = psd_v2hz[idx_t, :]

    # Remove duplicate times
    t_unique, t_uidx = np.unique(time, return_index=True)
    if len(t_unique) < len(time):
        time = t_unique
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

    psd_v2hz = psd_v2hz[:-1,:-1]

    return time, freq_mhz, psd_v2hz



###############################################################################
# 3) Read both RAD2 (top) and RAD1 (bottom)
###############################################################################
cmap = 'jet'

#rad2files = [r"C:\Users\osant\Desktop\solar_physics_projects\soler\wi_l2_wav_rad2_20240101_v01.cdf", r"C:\Users\osant\Desktop\solar_physics_projects\soler\wi_l2_wav_rad2_20240102_v01.cdf"]
#rad1files = [r"C:\Users\osant\Desktop\solar_physics_projects\soler\wi_l2_wav_rad1_20240101_v01.cdf", r"C:\Users\osant\Desktop\solar_physics_projects\soler\wi_l2_wav_rad1_20240102_v01.cdf"]

time_rad2_mpl, freq_rad2_mhz, psd_rad2_v2hz = read_wind_waves_cdf("rad2", "2021/04/19", "2021/04/21")
time_rad1_mpl, freq_rad1_mhz, psd_rad1_v2hz = read_wind_waves_cdf("rad1", "2021/04/19", "2021/04/21")

TT2, FF2 = np.meshgrid(time_rad2_mpl, freq_rad2_mhz, indexing='ij')
TT1, FF1 = np.meshgrid(time_rad1_mpl, freq_rad1_mhz, indexing='ij')

###############################################################################
# 5) Plot with TWO SUBPLOTS, no vertical separation (hspace=0),
#    but RAD2 is TOP, RAD1 is BOTTOM
###############################################################################
fig, (ax_top, ax_bottom) = plt.subplots(
    nrows=2, ncols=1,
    figsize=(10,6),
    dpi=150,
    sharex=True  # share time axis
)
# Remove vertical space between subplots
fig.subplots_adjust(left=0.08, right=0.93, top=0.92, bottom=0.12, hspace=0)


#Make the following auto-range if you are automating things
vmin, vmax = 1e-15, 1e-9  # Adjust to your data range
log_norm = colors.LogNorm(vmin=vmin, vmax=vmax)

# --- RAD2 subplot (TOP) ---
mesh2 = ax_top.pcolormesh(
    TT2, FF2,
    psd_rad2_v2hz,
    shading='flat',
    cmap=cmap,
    norm=log_norm
)
ax_top.set_yscale('log')
ax_top.set_ylabel("RAD2 [MHz]", fontsize=8)
ax_top.set_title("Wind/WAVES", fontsize=10)
ax_top.tick_params(axis='both', labelsize=8)

# --- RAD1 subplot (BOTTOM) ---
mesh1 = ax_bottom.pcolormesh(
    TT1, FF1,
    psd_rad1_v2hz,
    shading='flat',
    cmap=cmap,
    norm=log_norm
)
ax_bottom.set_yscale('log')
ax_bottom.set_ylabel("RAD1 [MHz]", fontsize=8)
ax_bottom.set_xlabel("Time (UTC)", fontsize=8)
ax_bottom.tick_params(axis='both', labelsize=8)

# Shared colorbar for both subplots on the right
cbar = fig.colorbar(mesh2, ax=[ax_top, ax_bottom], orientation="vertical", pad=0.02, extend='both')
cbar.set_label("PSD (V^2/Hz)", rotation=270, labelpad=12, fontsize=8)
cbar.ax.tick_params(labelsize=7)

cbar.cmap.set_under('gray') # nothing is grayed out since fillvals are removed

# Format time axis on the bottom subplot
locator = AutoDateLocator()
formatter = ConciseDateFormatter(locator)
ax_bottom.xaxis.set_major_locator(locator)
ax_bottom.xaxis.set_major_formatter(formatter)
for label in ax_bottom.get_xticklabels(which='major'):
    label.set(rotation=0, ha='center')

plt.show()