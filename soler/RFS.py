import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

import cdflib
import datetime
import cdaweb

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

###############################################################################
# Load LFR data
###############################################################################

lfr_files = cdaweb.cdaweb_download_fido(dataset='PSP_FLD_L3_RFS_LFR', startdate="2019/04/17", enddate="2019/04/19")


freq_lfr_2d  = cdflib.CDF(lfr_files[0]).varget("frequency_lfr_stokes")     # shape: (nTimeLFR, nFreqLFR)
freq_lfr = freq_lfr_2d[0, :]  # Convert 2D -> 1D
freq_lfr_mhz = freq_lfr / 1e6  # optional: Hz -> MHz
psd_lfr_sfu_all = np.zeros((1, len(freq_lfr)))  # this is just to get the right shape for appending
time_lfr_all = np.array([])


for file in lfr_files:
    cdf_lfr = cdflib.CDF(file)
    time_lfr_ns  = cdf_lfr.varget("epoch_lfr_stokes")         # shape: (nTimeLFR,)
    psd_lfr_sfu  = cdf_lfr.varget("psp_fld_l3_rfs_lfr_PSD_SFU") # shape: (nTimeLFR, nFreqLFR)
    # cdf_lfr.close()  # I think this is outdated

    # time_lfr_dt  = j2000_ns_to_datetime(time_lfr_ns)
    time_lfr_dt  = cdflib.epochs.CDFepoch.to_datetime(time_lfr_ns)
    time_lfr_mpl = mdates.date2num(time_lfr_dt)
    
    time_lfr_all = np.append(time_lfr_all, time_lfr_mpl)
    psd_lfr_sfu_all = np.append(psd_lfr_sfu_all, psd_lfr_sfu, axis=0)

psd_lfr_sfu_all = psd_lfr_sfu_all[1:,:]     # remove the zero row

###############################################################################
# Load HFR data
###############################################################################
# hfr_file = "/Users/ijebaraj/Downloads/Post_doc_Turku/example_GPT/psp_fld_l3_rfs_hfr_20190417_v03.cdf"  # <-- update

hfr_files = cdaweb.cdaweb_download_fido(dataset='PSP_FLD_L3_RFS_HFR', startdate="2019/04/17", enddate="2019/04/19")

freq_hfr_2d  = cdflib.CDF(hfr_files[0]).varget("frequency_hfr_stokes")     # shape: (nTimeHFR, nFreqHFR)
freq_hfr = freq_hfr_2d[0, :]  # Convert 2D -> 1D
freq_hfr_mhz = freq_hfr / 1e6  # optional: Hz -> MHz

time_hfr_all = np.array([])
psd_hfr_sfu_all = np.zeros((1, len(freq_hfr)))


for file in hfr_files:
    cdf_hfr = cdflib.CDF(file)
    time_hfr_ns  = cdf_hfr.varget("epoch_hfr_stokes")         # shape: (nTimeHFR,)
    psd_hfr_sfu  = cdf_hfr.varget("psp_fld_l3_rfs_hfr_PSD_SFU") # shape: (nTimeHFR, nFreqHFR)
    
    # cdf_hfr.close()  # I think this is outdated
    
    # time_hfr_dt  = j2000_ns_to_datetime(time_hfr_ns)
    time_hfr_dt  = cdflib.epochs.CDFepoch.to_datetime(time_hfr_ns)
    time_hfr_mpl = mdates.date2num(time_hfr_dt)
    
    time_hfr_all = np.append(time_hfr_all, time_hfr_mpl)
    psd_hfr_sfu_all = np.append(psd_hfr_sfu_all, psd_hfr_sfu, axis=0)

psd_hfr_sfu_all = psd_hfr_sfu_all[1:,:]

###############################################################################
# Build meshes for pcolormesh
###############################################################################
TimeLFR2D, FreqLFR2D = np.meshgrid(time_lfr_all, freq_lfr_mhz, indexing='ij')
TimeHFR2D, FreqHFR2D = np.meshgrid(time_hfr_all, freq_hfr_mhz, indexing='ij')

TimeLFR2D, FreqLFR2D = np.meshgrid(time_lfr_all, freq_lfr_mhz, indexing='ij')
TimeHFR2D, FreqHFR2D = np.meshgrid(time_hfr_all, freq_hfr_mhz, indexing='ij')

###############################################################################
# Custom colormap: gray for data < vmin, then Spectral
###############################################################################
cmap_spectral = cm.get_cmap('Spectral', 256)
colors_combined = np.vstack((
    [0.5, 0.5, 0.5, 1.0], 
    cmap_spectral(np.linspace(0, 1, 256))
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
    psd_hfr_sfu_all,
    shading='nearest',
    cmap=custom_cmap,
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
    psd_lfr_sfu_all,
    shading='nearest',
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
cbar.cmap.set_under('gray')

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
# ax_lfr.set_xlim(dt.datetime(2019, 4, 17, 17, 0), dt.datetime(2019, 4, 17, 23, 59))

plt.show()