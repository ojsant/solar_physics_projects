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


class Options:
    
    def __init__(self):
        style = {'description_width' : '50%'}
        layout = w.Layout(width="20%")
        
        self.radio = w.Checkbox(value=True, description="SWAVES LFR/HFR", indent=False, style=style, layout=layout)
        self.electrons = w.Checkbox(value=True, description="Electrons", indent=False, style=style, layout=layout)
        self.het = w.Checkbox(value=True, description="HET", indent=False, style=style, layout=layout)
        self.protons = w.Checkbox(value=True, description="Protons", indent=False, style=style, layout=layout)
        self.pad = w.Checkbox(value=False, description="PAD (not implemented)", disabled=True, indent=False, style=style, layout=layout)    # TODO: remove disabled keyword after implementation
        self.mag = w.Checkbox(value=False, description="MAG", indent=False, style=style, layout=layout)
        self.mag_angles = w.Checkbox(value=True, description="MAG angles", indent=False, style=style, layout=layout)
        self.Vsw = w.Checkbox(value=True, description="V_sw", indent=False, style=style, layout=layout)
        self.N = w.Checkbox(value=True, description="N", indent=False, style=style, layout=layout)
        self.T = w.Checkbox(value=True, description="T", indent=False, style=style, layout=layout)
        self.polarity = w.Checkbox(value=True, description="Polarity", indent=False, style=style, layout=layout)
        self.startdate = w.NaiveDatetimePicker(value=None, disabled=False, min=dt.datetime(2006, 10, 26), max=dt.datetime.now() - dt.timedelta(hours=1), description="Start date (data):", style=style, layout=layout)
        self.enddate = w.NaiveDatetimePicker(value=None, disabled=False, min=dt.datetime(2006, 10, 26) + dt.timedelta(hours=1), max=dt.datetime.now(), description="End date (data):", style=style, layout=layout)
        self.sept_viewing = w.Dropdown(options=['sun', 'asun', 'north', 'south'], value='sun', description='SEPT viewing:', style=style, layout=layout)
        self.sc = w.Dropdown(options=['A', 'B'], value='A', description='Spacecraft (A or B):', style=style, layout=layout)
        self.n_sept_e = w.SelectMultiple(options=range(0,14+1), description='SEPT electron energies:', rows=10, style=style, layout=layout)
        self.n_sept_p = w.SelectMultiple(options=range(0,29+1), description='SEPT proton energies:', rows=10, style=style, layout=layout)
        self.n_het_p =  w.SelectMultiple(options=range(0,10+1), description='HET proton energies:', rows=10, style=style, layout=layout)
        self.legends_inside = w.Checkbox(value=False, description='Legends inside', indent=False, style=style, layout=layout)
        self.resample = w.BoundedIntText(value=15, min=0, max=30, step=1, description='Resampling (min):', disabled=False, style=style, layout=layout)
        self.resample_mag = w.BoundedIntText(value=5, min=0, max=30, step=1, description='MAG resampling (min):', disabled=False, style=style, layout=layout)
        self.resample_pol = w.BoundedIntText(value=1, min=0, max=30, step=1, description='Polarity resampling (min):', disabled=False, style=style, layout=layout)
        self.pos_timestamp = w.Dropdown(options=['center', 'start', 'original'], description='Timestamp position', style=style, layout=layout)
        self.path = None
        self.plot_range = None
        
    def disp(self):
        for attr in vars(self):
            if attr not in ["path", "plot_range"]:
                display(vars(self)[attr])

    # for testing purposes
    def set_test_values(self):
        self.radio.value = True
        self.electrons.value  = True
        self.Vsw.value = True
        self.N.value = True
        self.startdate.value = dt.datetime(2022,3,14,00)
        self.enddate.value = dt.datetime(2022,3,16,00)
        self.n_het_p.value = (4, 7, 8, 9)
        self.n_sept_p.value = (1, 2, 3, 7, 8, 9, 10, 11, 19, 20, 21)
        self.n_sept_e.value = (3, 4, 5, 6, 7, 8, 12, 13, 14)
        

def plot_range_interval(startdate, enddate):
    timestamps = []
    date_iter = startdate
    while date_iter <= enddate:
        timestamps.append(date_iter)
        date_iter = date_iter + dt.timedelta(hours=1)
    return timestamps


def date_selector(startdate=None, enddate=None):
    """
    An HTML work-around to display datetimes without clipping on SelectionRangeSlider readouts.

    Author: Marcus Reaiche (https://github.com/jupyter-widgets/ipywidgets/issues/2855#issuecomment-966747483)
    """
    
    # Define date range
    if startdate is None and enddate is None:
        dates = plot_range_interval(startdate=options.startdate.value, enddate=options.enddate.value)

    else:
        dates = plot_range_interval(startdate=startdate, enddate=enddate)

    # First and last dates are selected by default
    initial_selection = (0, len(dates) - 1)

    # Define the date range slider: set readout to False
    date_range_selector = w.SelectionRangeSlider(
        options=dates,
        description="Plot range",
        index=initial_selection,
        continous_update=False,
        readout=False
    )

    # Define the display to substitute the readout
    date_range_display = w.HTML(
        value=(
            f"<b>{dates[initial_selection[0]]}" + 
            f" - {dates[initial_selection[1]]}</b>"))

    # Define the date range using the widgets.HBox
    date_range = w.HBox(
        (date_range_selector, date_range_display))

    # Callback function that updates the display
    def callback(dts):
        date_range_display.value = f"<b>{dts[0]} - {dts[1]}</b>"

    w.interactive_output(
        callback, 
        {"dts": date_range_selector})
    
    options.plot_range = date_range
    return date_range




def load_swaves(dataset, startdate, enddate, path=None):
    """
    Load STEREO/WAVES data from CDAWeb.

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
        print(f"No SWAVES radio data found between {startdate} and {enddate}")
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


def load_data():
    global df_sept_electrons_orig
    global df_sept_protons_orig
    global df_het_orig
    global df_waves_hfr
    global df_waves_lfr
    global df_sept_electrons
    global df_sept_protons
    global df_het
    global df_mag
    global df_magplas
    global meta_se
    global meta_sp
    global meta_het
    global df_magplas_pol

    global startdate
    global enddate
    global sept_viewing
    global pos_timestamp
    global path
    global sc
    global plot_radio
    global plot_mag
    global plot_Vsw
    global plot_N
    global plot_T
    global plot_het
    global plot_polarity

    startdate = options.startdate.value
    enddate = options.enddate.value
    sept_viewing = options.sept_viewing.value
    pos_timestamp = options.pos_timestamp.value
    path = options.path
    sc = options.sc.value
    plot_radio = options.radio.value
    plot_mag = options.mag.value
    plot_Vsw = options.Vsw.value
    plot_N = options.N.value
    plot_T = options.T.value
    plot_het = options.het.value
    plot_polarity = options.polarity.value

    resample = str(options.resample.value) + "min"         # convert to form that Pandas accepts
    resample_mag = str(options.resample_mag.value) + "min"
    resample_pol = str(options.resample_pol.value) + "min"

    df_sept_electrons_orig, meta_se = stereo_load(instrument='SEPT', startdate=startdate, enddate=enddate, 
                        sept_species='e', sept_viewing=sept_viewing,
                        path=path, pos_timestamp=pos_timestamp, spacecraft=sc)
    df_sept_protons_orig, meta_sp = stereo_load(instrument='SEPT', startdate=startdate, enddate=enddate, 
                            sept_species='p', sept_viewing=sept_viewing,
                            path=path, pos_timestamp=pos_timestamp, spacecraft=sc)
    
    if plot_het:
        df_het_orig, meta_het = stereo_load(instrument='HET', startdate=startdate, enddate=enddate,
                        path=path, pos_timestamp=pos_timestamp, spacecraft=sc)

    if plot_mag:
        df_mag_orig, meta_mag = stereo_load(spacecraft=sc, instrument='MAG', startdate=startdate, enddate=enddate, mag_coord='RTN', 
                                        path=path)

    if plot_Vsw or plot_N or plot_T:
        df_magplasma, meta_magplas = stereo_load(instrument='MAGPLASMA', startdate=startdate, enddate=enddate, 
                            path=path, pos_timestamp=pos_timestamp, spacecraft=sc)
      

    if plot_radio:
        df_waves_hfr = load_swaves("STA_L3_WAV_HFR", startdate=startdate, enddate=enddate, path=path)
        df_waves_lfr = load_swaves("STA_L3_WAV_LFR", startdate=startdate, enddate=enddate, path=path)

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



def make_plot():
    font_ylabel = 20
    font_legend = 10
    
    plot_electrons = options.electrons.value
    plot_protons = options.protons.value
    plot_mag_angles = options.mag_angles.value
    
    n_sept_e = options.n_sept_e.value
    n_sept_p = options.n_sept_p.value
    n_het_p = options.n_het_p.value
    n_het_e = range(0,2+1,1)

    legends_inside = options.legends_inside.value

    t_start = options.plot_range.children[0].value[0]
    t_end = options.plot_range.children[0].value[1]

    print(f"Chosen plot range:\n{t_start} - {t_end}")

    # #Channels list
    # channels_n_sept_e = range(0,14+1,n_sept_e)  # changed from np.arange()
    # channels_n_het_e = range(0,2+1,1)
    # channels_n_sept_p = range(0,29+1,n_sept_p)
    # channels_n_het_p = range(0,10+1,n_het_p)

    # channels_list = [channels_n_sept_e, channels_n_het_e, channels_n_sept_p, channels_n_het_p]

    #Chosen channels
    print('Chosen channels:')
    print(f'SEPT electrons: {n_sept_e}, {len(n_sept_e)}')
    print(f'HET electrons: {n_het_e}, {len(n_het_e)}')
    print(f'SEPT protons: {n_sept_p}, {len(n_sept_p)}')
    print(f'HET protons: {n_het_p}, {len(n_het_p)}')

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
        axs[i].set_prop_cycle('color', plt.cm.Reds_r(np.linspace(0,1,len(n_sept_e)+color_offset)))
        for channel in n_sept_e:
            axs[i].plot(df_sept_electrons.index, df_sept_electrons[f'ch_{channel+2}'],
                        ds="steps-mid", label='SEPT '+meta_se.ch_strings[channel+2])
        if plot_het:
            # plot het electron channels
            axs[i].set_prop_cycle('color', plt.cm.PuRd_r(np.linspace(0,1,4+color_offset)))
            for channel in n_het_e:
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
        num_channels = len(n_sept_p)# + len(n_het_p)
        axs[i].set_prop_cycle('color', plt.cm.plasma(np.linspace(0,1,num_channels+color_offset)))
        for channel in n_sept_p:
            axs[i].plot(df_sept_protons.index, df_sept_protons[f'ch_{channel+2}'], 
                    label='SEPT '+meta_sp.ch_strings[channel+2], ds="steps-mid")
        
        color_offset = 0 
        if plot_het:
            # plot het proton channels
            axs[i].set_prop_cycle('color', plt.cm.YlOrRd(np.linspace(0.2,1,len(n_het_p)+color_offset)))
            for channel in n_het_p:
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

options = Options()
