# from IPython.core.display import display, HTML
# display(HTML(data="""<style> div#notebook-container { width: 80%; } div#menubar-container { width: 85%; } div#maintoolbar-container { width: 90%; } </style>"""))
import numpy as np
import os
import pandas as pd
import warnings
import math
import cdflib
import sys
import sunpy

from matplotlib import pyplot as plt
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rcParams['agg.path.chunksize'] = 20000

from matplotlib import cm
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, Normalize

from datetime import timedelta
import datetime as dt

from seppy.loader.stereo import stereo_load
from seppy.util import resample_df

from sunpy.coordinates import get_horizons_coord
from sunpy.coordinates import frames

import ipywidgets as w
from IPython.core.display import display

from tools.my_func_py3 import mag_angles
from tools.polarity_plotting import polarity_rtn, polarity_panel, polarity_colorwheel
import tools.cdaweb as cdaweb


# omit Pandas' PerformanceWarning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action='ignore', message='No units provided for variable', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', message='astropy did not recognize units of', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')


def load_swaves(dataset, startdate, enddate, path=None):
    """
    Load STA WAVES data from CDAWeb.

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

    if len(files) == 0:
        print(f"No WAVES radio data found between {startdate} and {enddate}")
        return pd.DataFrame()
    else:
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



def load_data(opt):

    df_sept_electrons_orig = None
    df_sept_protons_orig = None
    df_het_orig = None
    df_waves_hfr = None
    df_waves_lfr = None
    df_sept_electrons = None
    df_sept_protons = None
    df_het = None
    df_mag = None
    df_magplas = None
    meta_se = None
    meta_sp = None
    meta_het = None
    df_magplasma = None

    startdate = opt["startdate"]
    enddate = opt["enddate"]
    sept_viewing = opt["sept_viewing"]
    pos_timestamp = opt["pos_timestamp"]
    file_path = opt["file_path"]
    sc = opt["sc"]
    plot_radio = opt["Radio"].value
    plot_mag = opt["Mag"].value
    plot_Vsw = opt["V_sw"].value
    plot_N = opt["N"].value
    plot_T = opt["T"].value
    plot_het = opt["HET"].value
    plot_polarity = opt["Polarity"].value

    resample = opt["resample"]
    resample_mag = opt["resample_mag"]
    resample_pol = opt["resample_pol"]

    df_sept_electrons_orig, meta_se = stereo_load(instrument='SEPT', startdate=startdate, enddate=enddate, 
                        sept_species='e', sept_viewing=sept_viewing,
                        path=file_path, pos_timestamp=pos_timestamp, spacecraft=sc)
    df_sept_protons_orig, meta_sp = stereo_load(instrument='SEPT', startdate=startdate, enddate=enddate, 
                            sept_species='p', sept_viewing=sept_viewing,
                            path=file_path, pos_timestamp=pos_timestamp, spacecraft=sc)
    
    if plot_het:
        df_het_orig, meta_het = stereo_load(instrument='HET', startdate=startdate, enddate=enddate,
                        path=file_path, pos_timestamp=pos_timestamp, spacecraft=sc)

    if plot_mag:
        df_mag_orig, meta_mag = stereo_load(spacecraft=sc, instrument='MAG', startdate=startdate, enddate=enddate, mag_coord='RTN', 
                                        path=file_path)

    if plot_Vsw or plot_N or plot_T:
        df_magplasma, meta_magplas = stereo_load(instrument='MAGPLASMA', startdate=startdate, enddate=enddate, 
                            path=file_path, pos_timestamp=pos_timestamp, spacecraft=sc)
      

    if plot_radio:
        df_waves_hfr = load_swaves("STA_L3_WAV_HFR", startdate=startdate, enddate=enddate, path=file_path)
        df_waves_lfr = load_swaves("STA_L3_WAV_LFR", startdate=startdate, enddate=enddate, path=file_path)

    if resample is not None:
        df_sept_electrons = resample_df(df_sept_electrons_orig, resample)  
        df_sept_protons = resample_df(df_sept_protons_orig, resample)  
        if plot_het:
            df_het  = resample_df(df_het_orig, resample)  
        if plot_Vsw or plot_N or plot_T:
            df_magplas = resample_df(df_magplasma, resample_mag) 
        if plot_mag:
            df_mag = resample_df(df_mag_orig, resample_mag)
        if plot_polarity:
            df_magplas_pol = resample_df(df_magplasma, resample_pol)
    
    else:
        df_sept_electrons = df_sept_electrons_orig
        df_sept_protons = df_sept_protons_orig  
        if plot_het:
            df_het  = df_het_orig
        if plot_Vsw or plot_N or plot_T:
            df_magplas = df_magplasma
        if plot_mag:
            df_mag = df_mag_orig
        if plot_polarity:
            df_magplas_pol = df_magplasma

    data = {}
    data["meta_se"] = meta_se
    data["meta_sp"] = meta_sp
    data["meta_het"] = meta_het
    
    data["df_waves_hfr"] = df_waves_hfr
    data["df_waves_lfr"] = df_waves_lfr
    data["df_sept_electrons"] = df_sept_electrons
    data["df_sept_protons"] = df_sept_protons
    data["df_het"] = df_het
    data["df_mag"] = df_mag
    data["df_magplas"] = df_magplas
    data["df_magplas_pol"] = df_magplas_pol
 
    return data

def make_plot(data, opt):
    font_ylabel = 20
    font_legend = 10

    meta_se = data["meta_se"] 
    meta_sp = data["meta_sp"] 
    meta_het = data["meta_het"] 
    
    df_waves_hfr = data["df_waves_hfr"] 
    df_waves_lfr = data["df_waves_lfr"] 
    df_sept_electrons = data["df_sept_electrons"] 
    df_sept_protons = data["df_sept_protons"] 
    df_het = data["df_het"] 
    df_mag = data["df_mag"] 
    df_magplas = data["df_magplas"]
    df_magplas_pol = data["df_magplas_pol"]

    startdate = opt["startdate"]
    enddate = opt["enddate"]
    sept_viewing = opt["sept_viewing"]
    
    sc = opt["sc"]

    plot_electrons = opt["Electrons"].value
    plot_protons = opt["Protons"].value
    plot_pad = opt["PAD"].value
    plot_mag_angles = opt["Mag angles"].value
    plot_polarity = opt["Polarity"].value
    plot_radio = opt["Radio"].value
    plot_mag = opt["Mag"].value
    plot_Vsw = opt["V_sw"].value
    plot_N = opt["N"].value
    plot_T = opt["T"].value
    plot_het = opt["HET"].value
    n_sept_e = opt["n_sept_e"]
    n_sept_p = opt["n_sept_p"]
    n_het_p = opt["n_het_p"]

    legends_inside = opt["legends_inside"]

    t_start = opt["plot_start"]
    t_end = opt["plot_end"]

    #Channels list
    channels_n_sept_e = list(np.arange(0,14+1,n_sept_e)) # why not just range()??
    channels_n_het_e = list(np.arange(0,2+1,1))
    channels_n_sept_p = list(np.arange(0,29+1,n_sept_p))
    channels_n_het_p = list(np.arange(0,10+1,n_het_p))

    channels_list = [channels_n_sept_e, channels_n_het_e, channels_n_sept_p, channels_n_het_p]

    #Chosen channels
    print('Channels:')
    print('sept_e:',channels_list[0],',', len(channels_list[0]))
    print('het_e:',channels_list[1],',', len(channels_list[1]))
    print('sept_p:',channels_list[2],',', len(channels_list[2]))
    print('het_p:',channels_list[3],',', len(channels_list[3]))

    panels = 1*plot_radio + 1*plot_electrons + 1*plot_protons  + 2*plot_mag_angles + 1*plot_mag + 1* plot_Vsw + 1* plot_N + 1* plot_T # + 1*plot_pad

    panel_ratios = list(np.zeros(panels)+1)

    if plot_radio:
        panel_ratios[0] = 2
    if plot_electrons and plot_protons:
        panel_ratios[0+1*plot_radio] = 2
        panel_ratios[1+1*plot_radio] = 2
    if plot_electrons or plot_protons:    
        panel_ratios[0+1*plot_radio] = 2

    if panels == 3:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 4*panels])#, gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")
    else:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 3*panels], gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")
    fig.subplots_adjust(hspace=0.1)

    i = 0


    color_offset = 4
    if plot_radio:
        vmin, vmax = 500, 1e7
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        
        TimeHFR2D, FreqHFR2D = np.meshgrid(df_waves_hfr.index, df_waves_hfr.columns, indexing='ij')
        TimeLFR2D, FreqLFR2D = np.meshgrid(df_waves_lfr.index, df_waves_lfr.columns, indexing='ij')

        # Create colormeshes. Shading option flat and thus the removal of last row and column are there to solve the time jump bar problem, 
        # when resampling isn't used
        mesh = axs[i].pcolormesh(TimeLFR2D, FreqLFR2D, df_waves_lfr.iloc[:-1,:-1], shading='flat', cmap='jet', norm=log_norm)
        axs[i].pcolormesh(TimeHFR2D, FreqHFR2D, df_waves_hfr.iloc[:-1,:-1], shading='flat', cmap='jet', norm=log_norm) # TODO: check if on top

        axs[i].set_yscale('log')
        axs[i].set_ylabel("Frequency (MHz)", fontsize=font_ylabel)
        
        # Add inset axes for colorbar
        axins = inset_axes(axs[i], width="100%", height="100%", loc="center", bbox_to_anchor=(1.05,0,0.03,1), bbox_transform=axs[i].transAxes, borderpad=0.2)
        cbar = fig.colorbar(mesh, cax=axins, orientation="vertical")
        cbar.set_label("Intensity (sfu)", rotation=90, labelpad=10, fontsize=font_ylabel)
        i += 1

    if plot_electrons:
        # plot sept electron channels
        axs[i].set_prop_cycle('color', plt.cm.Reds_r(np.linspace(0,1,len(channels_list[0])+color_offset)))
        for channel in channels_list[0]:
            axs[i].plot(df_sept_electrons.index, df_sept_electrons[f'ch_{channel+2}'],
                        ds="steps-mid", label='SEPT '+meta_se.ch_strings[channel+2])
        if plot_het:
            # plot het electron channels
            axs[i].set_prop_cycle('color', plt.cm.PuRd_r(np.linspace(0,1,4+color_offset)))
            for channel in channels_list[1]:
                axs[i].plot(df_het[f'Electron_Flux_{channel}'], 
                            label='HET '+meta_het['channels_dict_df_e'].ch_strings[channel],
                        ds="steps-mid")
        
        axs[i].set_ylabel("Flux\n"+r"[(cm$^2$ sr s MeV)$^{-1}]$", fontsize=font_ylabel)
        if legends_inside:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                    title=f'Electrons (SEPT: {sept_viewing}, HET: sun)', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
                    title=f'Electrons (SEPT: {sept_viewing}, HET: sun)', fontsize=font_legend)
        axs[i].set_yscale('log')
        i +=1    

        
    color_offset = 2    
    if plot_protons:
        # plot sept proton channels
        num_channels = len(channels_list[2])# + len(channels_list[3])
        axs[i].set_prop_cycle('color', plt.cm.plasma(np.linspace(0,1,num_channels+color_offset)))
        for channel in channels_list[2]:
            axs[i].plot(df_sept_protons.index, df_sept_protons[f'ch_{channel+2}'], 
                    label='SEPT '+meta_sp.ch_strings[channel+2], ds="steps-mid")
        
        color_offset = 0 
        if plot_het:
            # plot het proton channels
            axs[i].set_prop_cycle('color', plt.cm.YlOrRd(np.linspace(0.2,1,len(channels_list[3])+color_offset)))
            for channel in channels_list[3]:
                axs[i].plot(df_het.index, df_het[f'Proton_Flux_{channel}'], 
                        label='HET '+meta_het['channels_dict_df_p'].ch_strings[channel], ds="steps-mid")
        
        axs[i].set_ylabel("Flux\n"+r"[(cm$^2$ sr s MeV)$^{-1}]$", fontsize=font_ylabel)
        if legends_inside:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                    title=f'Ions (SEPT: {sept_viewing}, HET: sun)', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
                    title=f'Ions (SEPT: {sept_viewing}, HET: sun)', fontsize=font_legend)
        axs[i].set_yscale('log')
        i +=1    
        
    # plot magnetic field
    if plot_mag:
        ax = axs[i]
        ax.plot(df_mag.index, df_mag.BFIELD_3, label='B', color='k', linewidth=1)
        ax.plot(df_mag.index.values, df_mag.BFIELD_0.values, label='Br', color='dodgerblue')
        ax.plot(df_mag.index.values, df_mag.BFIELD_1.values, label='Bt', color='limegreen')
        ax.plot(df_mag.index.values, df_mag.BFIELD_2.values, label='Bn', color='deeppink')
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        if legends_inside:
            ax.legend(loc='upper right')
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            
        ax.set_ylabel('B [nT]', fontsize=font_ylabel)
        ax.tick_params(axis="x", direction="in", which='both')#, pad=-15)
        i += 1
        
    if plot_polarity:
        pos = get_horizons_coord(f'STEREO-{sc}', time={'start':df_magplas_pol.index[0]-pd.Timedelta(minutes=15),'stop':df_magplas_pol.index[-1]+pd.Timedelta(minutes=15),'step':"1min"})  # (lon, lat, radius) in (deg, deg, AU)
        pos = pos.transform_to(frames.HeliographicStonyhurst())
        #Interpolate position data to magnetic field data cadence
        r = np.interp([t.timestamp() for t in df_magplas_pol.index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.radius.value)
        lat = np.interp([t.timestamp() for t in df_magplas_pol.index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.lat.value)
        pol, phi_relative = polarity_rtn(df_magplas_pol.BFIELDRTN_0.values, df_magplas_pol.BFIELDRTN_1.values, df_magplas_pol.BFIELDRTN_2.values,r,lat,V=400)
        # create an inset axe in the current axe:
        pol_ax = inset_axes(ax, height="5%", width="100%", loc=9, bbox_to_anchor=(0.,0,1,1.1), bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
        pol_ax.get_xaxis().set_visible(False)
        pol_ax.get_yaxis().set_visible(False)
        pol_ax.set_ylim(0,1)
        pol_ax.set_xlim([df_magplas.index.values[0], df_magplas.index.values[-1]])
        pol_arr = np.zeros(len(pol))+1
        timestamp = df_magplas.index.values[2] - df_magplas.index.values[1]
        norm = Normalize(vmin=0, vmax=180, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        pol_ax.bar(df_magplas_pol.index.values[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
        pol_ax.bar(df_magplas_pol.index.values[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
        pol_ax.set_xlim(t_start, t_end)
        
    if plot_mag_angles:
        ax = axs[i]
        #Bmag = np.sqrt(np.nansum((mag_data.B_r.values**2,mag_data.B_t.values**2,mag_data.B_n.values**2), axis=0))    
        alpha, phi = mag_angles(df_mag.BFIELD_3, df_mag.BFIELD_0.values, df_mag.BFIELD_1.values,
                                df_mag.BFIELD_2.values)
        ax.plot(df_mag.index, alpha, '.k', label='alpha', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-90, 90)
        ax.set_ylabel(r"$\Theta_\mathrm{B}$ [°]", fontsize=font_ylabel)
        # ax.set_xlim(X1, X2)
        ax.tick_params(axis="x",direction="in", pad=-15)

        i += 1
        ax = axs[i]
        ax.plot(df_mag.index, phi, '.k', label='phi', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-180, 180)
        ax.set_ylabel(r"$\Phi_\mathrm{B}$ [°]", fontsize=font_ylabel)
        # ax.set_xlim(X1, X2)
        ax.tick_params(axis="x",direction="in", which='both', pad=-15)
        i += 1
        
    ### Temperature
    if plot_T:
        axs[i].plot(df_magplas.index, df_magplas['Tp'], '-k', label="Temperature")
        axs[i].set_ylabel(r"T$_\mathrm{p}$ [K]", fontsize=font_ylabel)
        axs[i].set_yscale('log')
        i += 1

    ### Density
    if plot_N:
        axs[i].plot(df_magplas.index, df_magplas.Np,
                    '-k', label="Ion density")
        axs[i].set_ylabel(r"N$_\mathrm{p}$ [cm$^{-3}$]", fontsize=font_ylabel)
        i += 1

    ### Sws
    if plot_Vsw:
        axs[i].plot(df_magplas.index, df_magplas.Vp,
                    '-k', label="Bulk speed")
        axs[i].set_ylabel(r"V$_\mathrm{sw}$ [kms$^{-1}$]", fontsize=font_ylabel)
        #i += 1
        
    axs[0].set_title(f'STEREO {sc}', ha='center')

    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%b %d'))
    axs[-1].xaxis.set_tick_params(rotation=0)
    axs[-1].set_xlabel(f"Time (UTC) / Date in {t_start.year}", fontsize=15)
    axs[-1].set_xlim(t_start, t_end)

    #plt.tight_layout()
    #fig.set_size_inches(12,15)
    fig.patch.set_facecolor('white')
    fig.set_dpi(200)
    plt.show()

    return fig, axs


def selection():
    options = ["Radio", "Electrons", "Protons", "PAD", "MAG angles", "MAG", "V_sw", "N", "T", "Polarity", "HET"]
    checkboxes = dict(zip(options, [w.Checkbox(value=False, description=option, indent=False) for option in options]))
    for option in options:
        display(checkboxes[option])
    
    return checkboxes

def plot_range_selection(opt):
    
    data_start = opt["startdate"]
    data_end = opt["enddate"]
    
    plot_start = w.NaiveDatetimePicker(description="Plot range start", value=data_start, min=data_start, max=data_end, disabled=False)
    plot_end = w.NaiveDatetimePicker(description="Plot range end", value=data_end, min=data_start, max=data_end, disabled=False)

    opt["plot_start"] = plot_start.value
    opt["plot_end"] = plot_end.value
    
    display(plot_start)
    display(plot_end)

    
    
# startdate = opt["startdate"]
# enddate = opt["enddate"]
# sept_viewing = opt["sept_viewing"]
# pos_timestamp = opt["pos_timestamp"]
# file_path = opt["file_path"]
# sc = opt["sc"]