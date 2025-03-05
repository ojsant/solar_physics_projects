# TODO:
# - figure out paths


# from IPython.core.display import display, HTML
# display(HTML(data="""<style> div#notebook-container { width: 80%; } div#menubar-container { width: 85%; } div#maintoolbar-container { width: 90%; } </style>"""))
from matplotlib.ticker import AutoMinorLocator#, LogLocator, NullFormatter, LinearLocator, MultipleLocator, 
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rcParams['agg.path.chunksize'] = 20000
import numpy as np
import os
import pandas as pd
import datetime as dt
import sunpy
import cdflib

from sunpy.coordinates import get_horizons_coord
from sunpy.coordinates import frames
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries
from sunpy.net import Scraper
from sunpy.time import TimeRange
from sunpy.data.data_manager.downloader import ParfiveDownloader

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm

from soler.tools.my_func_py3 import mag_angles, polarity_rtn, resample_df
from soler.tools.polarity_plotting import polarity_rtn, polarity_panel, polarity_colorwheel

from seppy.loader.wind import wind3dp_load
from seppy.loader.soho import soho_load
# from other_loaders_py3 import wind_3dp_av_en  #, wind_mfi_loader, ERNE_HED_loader

import ipywidgets as w
from IPython.core.display import display

#intensity_label = 'Intensity\n/(s cm² sr MeV)'
intensity_label = 'Intensity\n'+r'[(s cm² sr MeV)$^{-1}$]'



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

def wind_mfi_loader(startdate, enddate):

    dataset = 'WI_H3-RTN_MFI'  # 'WI_H2_MFI'
    cda_dataset = a.cdaweb.Dataset(dataset)

    trange = a.Time(startdate, enddate)

    # path = path_loc+'wind/mfi/'  # you can define here where the original data files should be saved, see 2 lines below
    path = None
    result = Fido.search(trange, cda_dataset)
    downloaded_files = Fido.fetch(result, path=path)  # use Fido.fetch(result, path='/ThisIs/MyPath/to/Data/{file}') to use a specific local folder for saving data files
    downloaded_files.sort()

    # read in data files to Pandas Dataframe
    data = TimeSeries(downloaded_files, concatenate=True)
    df = data.to_dataframe()

    # wind_datetime = np.arange(concat_df.shape[0]) * datetime.timedelta(hours=1)
    # for i in range(concat_df.shape[0]):
    #     dt64=df.index[i]
    #     ts=(dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    #     wind_datetime[i]=datetime.datetime.utcfromtimestamp(ts)

    # df['BR'], df['BT'], df['BN'] = cs.cxform('GSE','RTN', wind_datetime, x=df['BGSE_0'], y=df['BGSE_1'], z=df['BGSE_2'])
    # df['B'] = np.sqrt(df['BGSE_0'].values**2+df['BGSE_1'].values**2 +df['BGSE_2'].values**2)
    df['B'] = np.sqrt(df['BRTN_0'].values**2+df['BRTN_1'].values**2 +df['BRTN_2'].values**2)
    # return concat_df
    return  df


def selection():
    options = ["ERNE", "EPHIN", "WIND", "Radio", "Electrons", "Protons", "Pad?", "Mag angles", "Mag", "V_sw", "N", "T", "Polarity"]
    boxes = dict(zip(options, [w.Checkbox(value=False, description=quant, indent=False) for quant in options]))
    for option in options:
        display(boxes[option])


def load_data(opt):
    data = {}

    # LOAD DATA
    ####################################################################
    if opt["wind"]:
        data["edic_"], data["meta_e"] = wind3dp_load(dataset="WI_SFSP_3DP",
                            startdate=opt["startdate"],
                            enddate=opt["enddate"],
                            resample=0,
                            multi_index=True,
                            path=None,
                            threshold=opt["wind_flux_thres"])
        data["pdic_"], data["meta_p"] = wind3dp_load(dataset="WI_SOSP_3DP",
                            startdate=opt["startdate"],
                            enddate=opt["enddate"],
                            resample=0,
                            multi_index=True,
                            path=None,
                            threshold=opt["wind_flux_thres"])
        
    if opt["plot_radio"]:
        data["df_wind_wav_rad2"] = load_waves_rad(dataset="RAD2", startdate=opt["startdate"], enddate=opt["enddate"], file_path=None)
        data["df_wind_wav_rad1"] = load_waves_rad(dataset="RAD1", startdate=opt["startdate"], enddate=opt["enddate"], file_path=None)


    if opt["plot_ephin"]:
        data["ephin_"], data["meta_ephin"] = soho_load(dataset="SOHO_COSTEP-EPHIN_L2-1MIN", startdate=opt["startdate"], enddate=opt["enddate"],
                        path=None, resample=None, pos_timestamp=None)


    data["mag_data_"] = wind_mfi_loader(opt["startdate"], opt["enddate"])
    data["mag_data"] = resample_df(data["mag_data_"], '1min')  # resampling to 1min for polarity plot

    if opt["erne"]:
        data["erne_p_"], data["meta_erne"] = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN", startdate=opt["startdate"], enddate=opt["enddate"],
                            path=None, resample=None, pos_timestamp=None)
        
    #product = a.cdaweb.Dataset('AC_K0_SWE')
    #product = a.cdaweb.Dataset('WI_PLSP_3DP')
    #product = a.cdaweb.Dataset('WI_PM_3DP')  
    if opt["plot_Vsw"] or opt["plot_N"] or opt["plot_T"]:
        product = a.cdaweb.Dataset('WI_K0_3DP')

        time = a.Time(opt["startdate"], opt["enddate"])
        result = Fido.search(time & product)
        files = Fido.fetch(result, path=None)
        sw_data = TimeSeries(files, concatenate=True)
        data["df_solwind"] = sw_data.to_dataframe()
        data["df_solwind"]['vsw'] = np.sqrt(data["df_solwind"]['ion_vel_0']**2 + data["df_solwind"]['ion_vel_1']**2 + data["df_solwind"]['ion_vel_2']**2)
    
    # AVERAGING
    if opt["av_mag"] is not None:
        data["mag_df"] = resample_df(data["mag_data"], opt["av_mag"])
        if opt["plot_Vsw"] or opt["plot_N"] or opt["plot_T"]:
            data["vsw_df"] = resample_df(data["df_solwind"], opt["av_mag"])
    else:
        data["mag_df"] = data["mag_data"]
        if opt["plot_Vsw"] or opt["plot_T"] or opt["plot_N"]:
            data["vsw_df"] = data["df_solwind"]
        
    if opt["av_sep"] is not None:
        data["edic"] = resample_df(data["edic_"], opt["av_sep"])
        data["pdic"] = resample_df(data["pdic_"], opt["av_sep"])
        if opt["plot_ephin"]:
            data["ephin"] = resample_df(data["ephin_"], opt["av_sep"])
        if opt["erne"]:
            data["erne_p"] = resample_df(data["erne_p_"], opt["av_erne"])
    else:
        data["edic"] = data["edic_"]
        data["pdic"] = data["pdic_"]
        if opt["plot_ephin"]:
            data["ephin"] = data["ephin_"]
        if opt["erne"]:
            data["erne_p"] = data["erne_p_"]
        
    # add particles, SWE 


    return data

def make_plot(data, opt):
    panels = 1*opt["plot_radio"] + 1*opt["plot_electrons"] + 1*opt["plot_protons"] + 1*opt["plot_pad"] + 2*opt["plot_mag_angles"] + 1*opt["plot_mag"] + 1* opt["plot_Vsw"] + 1* opt["plot_N"] + 1* opt["plot_T"]

    panel_ratios = list(np.zeros(panels)+1)
    if opt["plot_radio"]:
        panel_ratios[0] = 2
    if opt["plot_electrons"] and opt["plot_protons"]:
        panel_ratios[0+1*opt["plot_radio"]] = 2
        panel_ratios[1+1*opt["plot_radio"]] = 2
    if opt["plot_electrons"] or opt["plot_protons"]:    
        panel_ratios[0+1*opt["plot_radio"]] = 2


    
    # PLOT
    ####################################################################
    i=0
    if panels == 3:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 4*panels])#, gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")
    else:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 3*panels], gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")
        #fig, axs = plt.subplots(nrows=panels, sharex=True, dpi=100, figsize=[7, 1.5*panels], gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")

        
    fig.subplots_adjust(hspace=0.1)

    if panels == 1:
        axs = [axs, axs]

    FONT_YLABEL = 20
    FONT_LEGEND = 10
    COLOR_OFFSET = 3

    if opt["plot_radio"]:
        vmin, vmax = 1e-15, 1e-10
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        
        time_rad2_2D, freq_rad2_2D = np.meshgrid(data["df_wind_wav_rad2"].index, data["df_wind_wav_rad2"].columns, indexing='ij')
        time_rad1_2D, freq_rad1_2D = np.meshgrid(data["df_wind_wav_rad1"].index, data["df_wind_wav_rad1"].columns, indexing='ij')

        # Create colormeshes. Shading option flat and thus the removal of last row and column are there to solve the time jump bar problem, 
        # when resampling isn't used
        mesh = axs[i].pcolormesh(time_rad1_2D, freq_rad1_2D, data["df_wind_wav_rad1"].iloc[:-1,:-1], shading='flat', cmap='jet', norm=log_norm)
        axs[i].pcolormesh(time_rad2_2D, freq_rad2_2D, data["df_wind_wav_rad2"].iloc[:-1,:-1], shading='flat', cmap='jet', norm=log_norm)

        axs[i].set_yscale('log')
        axs[i].set_ylabel("Frequency (MHz)", fontsize=FONT_YLABEL)
        
        # Add inset axes for colorbar
        axins = inset_axes(axs[i], width="100%", height="100%", loc="center", bbox_to_anchor=(1.05,0,0.03,1), bbox_transform=axs[i].transAxes, borderpad=0.2)
        cbar = fig.colorbar(mesh, cax=axins, orientation="vertical")
        cbar.set_label(r"Intensity ($\mathrm{V^2/Hz}$)", rotation=90, labelpad=10, fontsize=FONT_YLABEL)
        i += 1

    if opt["plot_electrons"]:
        # electrons
        ax = axs[i]
        axs[i].set_prop_cycle('color', plt.cm.Greens_r(np.linspace(0,1, len(data["meta_e"]['channels_dict_df'])+COLOR_OFFSET)))
        if opt["wind"]:
            for ch in np.arange(1, len(data["meta_e"]['channels_dict_df'])):
                ax.plot(data["edic"].index, data["edic"][f'FLUX_{ch}'] * opt["wind_ev2MeV_fac"], label='Wind/3DP '+data["meta_e"]['channels_dict_df']['Bins_Text'].values[ch], drawstyle='steps-mid')
        
        
        if opt["plot_ephin"]:
            ax.plot(data["ephin"].index, data["ephin"][opt["eph_e_ch"]]*opt["intercal"], '-k', label='SOHO/EPHIN '+data["meta_ephin"][opt["eph_e_ch"]]+f' / {opt["intercal"]}', drawstyle='steps-mid')
        # ax.set_ylim(1e0, 1e4)
        ax.legend(title='Electrons', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONT_LEGEND)
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label, fontsize=FONT_YLABEL)
        i += 1
        
    COLOR_OFFSET = 2    
    if opt["plot_protons"]:    
        # protons low en:
        ax = axs[i]
        ax.set_prop_cycle('color', plt.cm.plasma(np.linspace(0,1, len(data["meta_p"]['channels_dict_df'])+COLOR_OFFSET)))
        if opt["wind"]:
            for ch in np.arange(2, len(data["meta_p"]['channels_dict_df'])):
                ax.plot(data["pdic"].index, data["pdic"][f'FLUX_{ch}'] * opt["wind_ev2MeV_fac"], label='Wind/3DP '+data["meta_p"]['channels_dict_df']['Bins_Text'].values[ch],
                        drawstyle='steps-mid')
        ax.legend(title='Protons', loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label, fontsize=FONT_YLABEL)

        # protons high en:
        if opt["erne"]:
            ax.set_prop_cycle('color', plt.cm.YlOrRd(np.linspace(0.2,1,10))) #cm.RdPu_r
            for ch in np.arange(0, 10):
                ax.plot(data["erne_p"].index, data["erne_p"][f'PH_{ch}'], label='SOHO/ERNE/HED '+data["meta_erne"]['channels_dict_df_p']['ch_strings'][ch], 
                            drawstyle='steps-mid')
        ax.legend(title='Protons', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONT_LEGEND)
        ax.set_yscale('log')
        i += 1

    
    if opt["plot_mag"]:    
        ax = axs[i]
        ax.plot(data["mag_df"].index, data["mag_df"].B.values, label='B', color='k', linewidth=1)
        ax.plot(data["mag_df"].index, data["mag_df"].BRTN_0.values, label='Br', color='dodgerblue', linewidth=1)
        ax.plot(data["mag_df"].index, data["mag_df"].BRTN_1.values, label='Bt', color='limegreen', linewidth=1)
        ax.plot(data["mag_df"].index, data["mag_df"].BRTN_2.values, label='Bn', color='deeppink', linewidth=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))#, title='RTN')#, bbox_to_anchor=(1, 0.5))
        ax.set_ylabel('B [nT]', fontsize=FONT_YLABEL)
        ax.tick_params(axis="x",direction="in", which='both') #, pad=-15
        i += 1
        
    if opt["plot_polarity"]:
        pos = get_horizons_coord('Wind', time={'start':data["mag_data"].index[0]-pd.Timedelta(minutes=15),
                                            'stop':data["mag_data"].index[-1]+pd.Timedelta(minutes=15),'step':"1min"}) 
                                                # (lon, lat, radius) in (deg, deg, AU)
        pos = pos.transform_to(frames.HeliographicStonyhurst())
        #Interpolate position data to magnetic field data cadence
        r = np.interp([t.timestamp() for t in data["mag_data"].index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.radius.value)
        lat = np.interp([t.timestamp() for t in data["mag_data"].index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.lat.value)
        pol, phi_relative = polarity_rtn(data["mag_data"].BRTN_0.values, data["mag_data"].BRTN_1.values, data["mag_data"].BRTN_2.values,r,lat,V=400)
    # create an inset axe in the current axe:
        pol_ax = inset_axes(ax, height="5%", width="100%", loc=9, bbox_to_anchor=(0.,0,1,1.1), bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
        pol_ax.get_xaxis().set_visible(False)
        pol_ax.get_yaxis().set_visible(False)
        pol_ax.set_ylim(0,1)
        pol_ax.set_xlim([data["mag_data"].index.values[0], data["mag_data"].index.values[-1]])
        pol_arr = np.zeros(len(pol))+1
        timestamp = data["mag_data"].index.values[2] - data["mag_data"].index.values[1]
        norm = Normalize(vmin=0, vmax=180, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        pol_ax.bar(data["mag_data"].index.values[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
        pol_ax.bar(data["mag_data"].index.values[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
        pol_ax.set_xlim(opt["startdate"], opt["enddate"])
        
        
    if opt["plot_mag_angles"]:
        alpha, phi = mag_angles(data["mag_df"].B.values, data["mag_df"].BRTN_0.values, data["mag_df"].BRTN_1.values, data["mag_df"].BRTN_2.values)
        ax = axs[i]
        ax.plot(data["mag_df"].index, alpha, '.k', label='alpha', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-90, 90)
        ax.set_ylabel(r"$\Theta_\mathrm{B}$ [°]", fontsize=FONT_YLABEL)
        ax.tick_params(axis="x",direction="in")
        i += 1
        
        ax = axs[i]
        ax.plot(data["mag_df"].index, phi, '.k', label='phi', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-180, 180)
        ax.set_ylabel(r"$\Phi_\mathrm{B}$ [°]", fontsize=FONT_YLABEL)
        ax.tick_params(axis="x",direction="in", which='both')
        i += 1
        
    ### Temperature
    if opt["plot_T"]:
        axs[i].plot(data["vsw_df"].index, data["vsw_df"]['ion_temp'], '-k', label="Temperature")
        axs[i].set_ylabel(r"T$_\mathrm{p}$ [K]", fontsize=FONT_YLABEL)
        i += 1

    ### Density
    if opt["plot_N"]:
        axs[i].plot(data["vsw_df"].index, data["vsw_df"].ion_density,
                    '-k', label="Ion density")
        axs[i].set_ylabel(r"N$_\mathrm{p}$ [cm$^{-3}$]", fontsize=FONT_YLABEL)
        i += 1

    ### Sws
    if opt["plot_Vsw"]:
        axs[i].plot(data["vsw_df"].index, data["vsw_df"].vsw,
                    '-k', label="Bulk speed")
        axs[i].set_ylabel(r"V$_\mathrm{sw}$ [km/s]", fontsize=FONT_YLABEL)
        i += 1
            

    axs[0].set_title('Near-Earth spacecraft (Wind, SOHO)', fontsize=FONT_YLABEL)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
    axs[-1].xaxis.set_tick_params(rotation=0)
    axs[-1].set_xlabel(f"Time (UTC) / Date in {opt["startdate"].year}", fontsize=15)
    axs[-1].set_xlim(opt["startdate"], opt["enddate"])


    plt.show()
    # if opt["save_fig"]:
    #     plt.savefig(f'{outpath}L1_multiplot_{str(opt["startdate"].date())}--{str(opt["enddate"].date())}_{opt["av_sep"]}.png')