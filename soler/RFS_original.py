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
# lfr_file = "/Users/ijebaraj/Downloads/Post_doc_Turku/example_GPT/psp_fld_l3_rfs_lfr_20190417_v03.cdf"  # <-- update
lfr_file = cdaweb.cdaweb_download_fido(dataset='PSP_FLD_L3_RFS_LFR', startdate="2019/04/17", enddate="2019/04/19")[0]

cdf_lfr = cdflib.CDF(lfr_file)

time_lfr_ns  = cdf_lfr.varget("epoch_lfr_stokes")          # shape: (nTimeLFR,)
freq_lfr_2d  = cdf_lfr.varget("frequency_lfr_stokes")      # shape: (nTimeLFR, nFreqLFR)
psd_lfr_sfu  = cdf_lfr.varget("psp_fld_l3_rfs_lfr_PSD_SFU") # shape: (nTimeLFR, nFreqLFR)
# cdf_lfr.close()  # I think this is outdated

freq_lfr = freq_lfr_2d[0, :]  # Convert 2D -> 1D if needed
# time_lfr_dt  = j2000_ns_to_datetime(time_lfr_ns)
time_lfr_dt  = cdflib.epochs.CDFepoch.to_datetime(time_lfr_ns)

time_lfr_mpl = mdates.date2num(time_lfr_dt)
freq_lfr_mhz = freq_lfr / 1e6  # optional: Hz -> MHz

psd_lfr_sfu = psd_lfr_sfu[:-1,:-1]
###############################################################################
# Load HFR data
###############################################################################
# hfr_file = "/Users/ijebaraj/Downloads/Post_doc_Turku/example_GPT/psp_fld_l3_rfs_hfr_20190417_v03.cdf"  # <-- update
hfr_file = cdaweb.cdaweb_download_fido(dataset='PSP_FLD_L3_RFS_HFR', startdate="2019/04/17", enddate="2019/04/19")[0]

cdf_hfr = cdflib.CDF(hfr_file)

time_hfr_ns  = cdf_hfr.varget("epoch_hfr_stokes")
freq_hfr_2d  = cdf_hfr.varget("frequency_hfr_stokes")
psd_hfr_sfu  = cdf_hfr.varget("psp_fld_l3_rfs_hfr_PSD_SFU")
# cdf_hfr.close()  # I think this is outdated
# for each time step except the last one:
for i in range(len(time_hfr_ns)-1):
    # check if time increases by more than 60s:
    if time_hfr_ns[i+1]-time_hfr_ns[i] > 60000000000:
        psd_hfr_sfu[i,:] = np.nan

freq_hfr = freq_hfr_2d[0, :]
# time_hfr_dt  = j2000_ns_to_datetime(time_hfr_ns)
time_hfr_dt = cdflib.epochs.CDFepoch.to_datetime(time_hfr_ns)
time_hfr_mpl = mdates.date2num(time_hfr_dt)
freq_hfr_mhz = freq_hfr / 1e6

psd_hfr_sfu = psd_hfr_sfu[:-1,:-1]

###############################################################################
# Build meshes for pcolormesh
###############################################################################
TimeLFR2D, FreqLFR2D = np.meshgrid(time_lfr_mpl, freq_lfr_mhz, indexing='ij')
TimeHFR2D, FreqHFR2D = np.meshgrid(time_hfr_mpl, freq_hfr_mhz, indexing='ij')

#TimeLFR2D, FreqLFR2D = np.meshgrid(time_lfr_dt, freq_lfr_mhz, indexing='ij')
#TimeHFR2D, FreqHFR2D = np.meshgrid(time_hfr_dt, freq_hfr_mhz, indexing='ij')

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
    psd_hfr_sfu,
    shading='auto',
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
    psd_lfr_sfu,
    shading='auto',
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