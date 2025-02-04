import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from scipy.interpolate import RegularGridInterpolator
import cdflib
import datetime

###############################################################################
# 1) Convert J2000 ns -> Python datetime
###############################################################################
def j2000_ns_to_datetime(ns_array):
    """
    Convert 'nanoseconds since 2000-01-01T12:00:00' (J2000)
    into a numpy array of Python datetime objects.
    """
    j2000_ref = datetime.datetime(2000, 1, 1, 12, 0, 0)
    return np.array([
        j2000_ref + datetime.timedelta(seconds=(ns * 1e-9))
        for ns in ns_array
    ])

###############################################################################
# 2) Read a single Wind/WAVES file (assuming freq is 1D or identical rows if 2D)
###############################################################################
def read_wind_waves_cdf(file_path, psd_var="PSD_V2_SP"):
    cdf = cdflib.CDF(file_path)
    
    # Time
    time_ns = cdf.varget("Epoch")  # shape (nTime,)
    time_dt = j2000_ns_to_datetime(time_ns)
    time_mpl = mdates.date2num(time_dt)
    
    # Frequency (1D or 2D with identical rows)
    freq_raw = cdf.varget("FREQUENCY")
    # PSD shape (nTime, nFreq)
    psd_raw = cdf.varget(psd_var)
    cdf.close()

    # If freq is 2D but each row is identical, take freq_raw[0,:]
    if freq_raw.ndim == 2:
        freq_hz = freq_raw[0, :]
    else:
        freq_hz = freq_raw

    # Some files use a fill value ~ -9.9999998e+30
    fill_val = -9.999999848243207e+30
    valid_mask = (freq_hz > 0) & (freq_hz != fill_val)
    freq_hz = freq_hz[valid_mask]
    psd_raw = psd_raw[:, valid_mask]

    # Convert frequency to MHz
    freq_mhz = freq_hz / 1e6

    # Sort time
    idx_t = np.argsort(time_mpl)
    time_mpl = time_mpl[idx_t]
    psd_raw  = psd_raw[idx_t, :]

    # Remove duplicate times
    t_unique, t_uidx = np.unique(time_mpl, return_index=True)
    if len(t_unique) < len(time_mpl):
        time_mpl = t_unique
        psd_raw  = psd_raw[t_uidx, :]

    # Sort freq
    idx_f = np.argsort(freq_mhz)
    freq_mhz = freq_mhz[idx_f]
    psd_raw  = psd_raw[:, idx_f]

    # Remove duplicate freqs
    f_unique, f_uidx = np.unique(freq_mhz, return_index=True)
    if len(f_unique) < len(freq_mhz):
        freq_mhz = f_unique
        psd_raw  = psd_raw[:, f_uidx]

    return time_mpl, freq_mhz, psd_raw

###############################################################################
# 3) Read both RAD2 (top) and RAD1 (bottom)
###############################################################################
rad2_file = "/Users/ijebaraj/Downloads/Post_doc_Turku/example_GPT/wi_l2_wav_rad2_20000101_v01.cdf"
rad1_file = "/Users/ijebaraj/Downloads/Post_doc_Turku/example_GPT/wi_l2_wav_rad1_20000101_v01.cdf"

time_rad2_mpl, freq_rad2_mhz, psd_rad2_v2hz = read_wind_waves_cdf(rad2_file, psd_var="PSD_V2_SP")
time_rad1_mpl, freq_rad1_mhz, psd_rad1_v2hz = read_wind_waves_cdf(rad1_file, psd_var="PSD_V2_SP")

###############################################################################
# 4) (Optional) Interpolate each dataset for smoother plotting
###############################################################################
upsample_factor = 2

def interpolate_2d(time_mpl, freq_mhz, psd_2d, factor=2):
    # Create new time/freq grids
    time_new = np.linspace(time_mpl[0], time_mpl[-1], factor * len(time_mpl))
    freq_new = np.linspace(freq_mhz[0], freq_mhz[-1], factor * len(freq_mhz))

    # Interpolator
    interp = RegularGridInterpolator(
        (time_mpl, freq_mhz),
        psd_2d,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    TT, FF = np.meshgrid(time_new, freq_new, indexing='ij')
    points = np.stack([TT.ravel(), FF.ravel()], axis=-1)
    psd_smooth = interp(points).reshape(TT.shape)

    return TT, FF, psd_smooth

TT2, FF2, psd_rad2_smooth = interpolate_2d(time_rad2_mpl, freq_rad2_mhz, psd_rad2_v2hz, factor=upsample_factor)
TT1, FF1, psd_rad1_smooth = interpolate_2d(time_rad1_mpl, freq_rad1_mhz, psd_rad1_v2hz, factor=upsample_factor)

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

# Define a custom colormap
cmap_spectral = cm.get_cmap('Spectral_r', 256)
colors_combined = np.vstack((
    [0.5, 0.5, 0.5, 1.0],  # color for <vmin
    cmap_spectral(np.linspace(0, 1, 256))
))
custom_cmap = ListedColormap(colors_combined)

#Make the following auto-range if you are automating things
vmin, vmax = 1e-15, 1e-9  # Adjust to your data range
log_norm = colors.LogNorm(vmin=vmin, vmax=vmax)

# --- RAD2 subplot (TOP) ---
mesh2 = ax_top.pcolormesh(
    TT2, FF2,
    psd_rad2_smooth,
    shading='auto',
    cmap=custom_cmap,
    norm=log_norm
)
ax_top.set_yscale('log')
ax_top.set_ylabel("RAD2 [MHz]", fontsize=8)
ax_top.set_title("Wind/WAVES", fontsize=10)
ax_top.tick_params(axis='both', labelsize=8)

# --- RAD1 subplot (BOTTOM) ---
mesh1 = ax_bottom.pcolormesh(
    TT1, FF1,
    psd_rad1_smooth,
    shading='auto',
    cmap=custom_cmap,
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
cbar.cmap.set_under('gray')

# Format time axis on the bottom subplot
locator = AutoDateLocator()
formatter = ConciseDateFormatter(locator)
ax_bottom.xaxis.set_major_locator(locator)
ax_bottom.xaxis.set_major_formatter(formatter)
for label in ax_bottom.get_xticklabels(which='major'):
    label.set(rotation=0, ha='center')

plt.show()