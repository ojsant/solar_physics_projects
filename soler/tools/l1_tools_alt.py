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

from tools.my_func_py3 import mag_angles, polarity_rtn, resample_df
from tools.polarity_plotting import polarity_rtn, polarity_panel, polarity_colorwheel

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
    ######### This is just for easier copy pasting from other versions (no need to change variables into dictionary references) #########
    df_wind_wav_rad2 = None
    df_wind_wav_rad1 = None
    df_mag_pol = None
    df_solwind = None
    df_mag = None
    df_vsw = None
    edic = None
    pdic = None
    ephin = None
    erne_p = None
    meta_ephin = None
    meta_erne = None
    meta_e = None
    meta_p = None

    data_path = None  # '/Users/dresing/data/projects/wind/'
    wind_data_path = None  # '/Users/jagies/data/wind/'
    erne_data_path = None  # '/Users/jagies/data/soho/'
    ephin_data_path = None  # '/Users/jagies/data/soho/'

    wind = opt["wind"]
    startdate = opt["startdate"]
    enddate = opt["enddate"]
    erne = opt["erne"]
    plot_ephin = opt["plot_ephin"]

    if plot_ephin:
        eph_e_ch = opt["eph_e_ch"]
        intercal = opt["intercal"]  #14.
        eph_p_ch = opt["eph_p_ch"]
    
    wind_ev2MeV_fac = opt["wind_ev2MeV_fac"]
    wind_flux_thres = opt["wind_flux_thres"]

    save_fig = opt["save_fig"]

    plot_radio = opt["plot_radio"]
    plot_electrons = opt["plot_electrons"]
    plot_protons = opt["plot_protons"]
    plot_pad = opt["plot_pad"]
    plot_mag_angles = opt["plot_mag_angles"] 
    plot_mag = opt["plot_mag"]
    plot_Vsw = opt["plot_Vsw"]
    plot_N = opt["plot_N"]
    plot_T = opt["plot_T"]
    plot_polarity = opt["plot_polarity"] 

    av_sep = opt["av_sep"]
    av_mag =  opt["av_mag"]
    av_erne = opt["av_erne"]

    # LOAD DATA
    ####################################################################
    if wind:
        edic_, meta_e = wind3dp_load(dataset="WI_SFSP_3DP",
                            startdate=startdate,
                            enddate=enddate,
                            resample=0,
                            multi_index=True,
                            path=wind_data_path,
                            threshold=wind_flux_thres)
        pdic_, meta_p = wind3dp_load(dataset="WI_SOSP_3DP",
                            startdate=startdate,
                            enddate=enddate,
                            resample=0,
                            multi_index=True,
                            path=wind_data_path,
                            threshold=wind_flux_thres)
        
    if plot_radio:
        df_wind_wav_rad2 = load_waves_rad(dataset="RAD2", startdate=startdate, enddate=enddate, file_path=None)
        df_wind_wav_rad1 = load_waves_rad(dataset="RAD1", startdate=startdate, enddate=enddate, file_path=None)


    if plot_ephin:
        ephin_, meta_ephin = soho_load(dataset="SOHO_COSTEP-EPHIN_L2-1MIN", startdate=startdate, enddate=enddate,
                        path=None, resample=None, pos_timestamp=None)


    mag_data = wind_mfi_loader(startdate, enddate)
    df_mag_pol = resample_df(mag_data, '1min')  # resampling to 1min for polarity plot

    if erne:
        erne_p_, meta_erne = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN", startdate=startdate, enddate=enddate,
                            path=None, resample=None, pos_timestamp=None)
        
    #product = a.cdaweb.Dataset('AC_K0_SWE')
    #product = a.cdaweb.Dataset('WI_PLSP_3DP')
    #product = a.cdaweb.Dataset('WI_PM_3DP')  
    if plot_Vsw or plot_N or plot_T:
        product = a.cdaweb.Dataset('WI_K0_3DP')

        time = a.Time(startdate, enddate)
        result = Fido.search(time & product)
        files = Fido.fetch(result, path=None)
        sw_data = TimeSeries(files, concatenate=True)
        df_solwind = sw_data.to_dataframe()
        df_solwind['vsw'] = np.sqrt(df_solwind['ion_vel_0']**2 + df_solwind['ion_vel_1']**2 + df_solwind['ion_vel_2']**2)
    
    # AVERAGING
    if av_mag is not None:
        df_mag = resample_df(mag_data, av_mag)
        if plot_Vsw or plot_N or plot_T:
            df_vsw = resample_df(df_solwind, av_mag)
    else:
        df_mag = mag_data
        if plot_Vsw or plot_T or plot_N:
            df_vsw = df_solwind
        
    if av_sep is not None:
        edic = resample_df(edic_, av_sep)
        pdic = resample_df(pdic_, av_sep)
        if plot_ephin:
            ephin = resample_df(ephin_, av_sep)
        if erne:
            erne_p = resample_df(erne_p_, av_erne)
    else:
        edic = edic_
        pdic = pdic_
        if plot_ephin:
            ephin = ephin_
        if erne:
            erne_p = erne_p_
        
    # add particles, SWE 


    # Gather everything into a dictionary
    data["df_wind_wav_rad2"] = df_wind_wav_rad2
    data["df_wind_wav_rad1"] = df_wind_wav_rad1
    data["df_mag_pol"] = df_mag_pol
    data["df_mag"] = df_mag
    data["df_vsw"] = df_vsw

    data["edic"] = edic
    data["pdic"] = pdic
    data["ephin"] = ephin
    data["erne_p"] = erne_p

    data["meta_ephin"] = meta_ephin
    data["meta_erne"] = meta_erne
    data["meta_e"] = meta_e
    data["meta_p"] = meta_p

    return data

def make_plot(data, opt):

    df_wind_wav_rad2 = data["df_wind_wav_rad2"]  
    df_wind_wav_rad1 = data["df_wind_wav_rad1"]  
    df_mag_pol = data["df_mag_pol"]
    df_mag = data["df_mag"]  
    df_vsw = data["df_vsw"]  

    edic = data["edic"]  
    pdic = data["pdic"]  
    ephin = data["ephin"]  
    erne_p = data["erne_p"]  

    meta_ephin =data["meta_ephin"]  
    meta_erne =data["meta_erne"]  
    meta_e = data["meta_e"]  
    meta_p = data["meta_p"]  

    wind = opt["wind"]
    startdate = opt["startdate"]
    enddate = opt["enddate"]
    erne = opt["erne"]
    plot_ephin = opt["plot_ephin"]

    if plot_ephin:
        eph_e_ch = opt["eph_e_ch"]
        intercal = opt["intercal"]  #14.
        eph_p_ch = opt["eph_p_ch"]
    
    wind_ev2MeV_fac = opt["wind_ev2MeV_fac"]

    plot_radio = opt["plot_radio"]
    plot_electrons = opt["plot_electrons"]
    plot_protons = opt["plot_protons"]
    plot_pad = opt["plot_pad"]
    plot_mag_angles = opt["plot_mag_angles"] 
    plot_mag = opt["plot_mag"]
    plot_Vsw = opt["plot_Vsw"]
    plot_N = opt["plot_N"]
    plot_T = opt["plot_T"]
    plot_polarity = opt["plot_polarity"] 


    FONT_YLABEL = 20
    FONT_LEGEND = 10

    panels = 1*plot_radio + 1*plot_electrons + 1*plot_protons + 1*plot_pad + 2*plot_mag_angles + 1*plot_mag + 1* plot_Vsw + 1* plot_N + 1* plot_T

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
        #fig, axs = plt.subplots(nrows=panels, sharex=True, dpi=100, figsize=[7, 1.5*panels], gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")

        
    fig.subplots_adjust(hspace=0.1)

    if panels == 1:
        axs = [axs, axs]

    
    color_offset = 3
    i = 0

    if plot_radio:
        vmin, vmax = 1e-15, 1e-10
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        
        time_rad2_2D, freq_rad2_2D = np.meshgrid(df_wind_wav_rad2.index, df_wind_wav_rad2.columns, indexing='ij')
        time_rad1_2D, freq_rad1_2D = np.meshgrid(df_wind_wav_rad1.index, df_wind_wav_rad1.columns, indexing='ij')

        # Create colormeshes. Shading option flat and thus the removal of last row and column are there to solve the time jump bar problem, 
        # when resampling isn't used
        mesh = axs[i].pcolormesh(time_rad1_2D, freq_rad1_2D, df_wind_wav_rad1.iloc[:-1,:-1], shading='flat', cmap='jet', norm=log_norm)
        axs[i].pcolormesh(time_rad2_2D, freq_rad2_2D, df_wind_wav_rad2.iloc[:-1,:-1], shading='flat', cmap='jet', norm=log_norm)

        axs[i].set_yscale('log')
        axs[i].set_ylabel("Frequency (MHz)", fontsize=FONT_YLABEL)
        
        # Add inset axes for colorbar
        axins = inset_axes(axs[i], width="100%", height="100%", loc="center", bbox_to_anchor=(1.05,0,0.03,1), bbox_transform=axs[i].transAxes, borderpad=0.2)
        cbar = fig.colorbar(mesh, cax=axins, orientation="vertical")
        cbar.set_label(r"Intensity ($\mathrm{V^2/Hz}$)", rotation=90, labelpad=10, fontsize=FONT_YLABEL)
        i += 1

    if plot_electrons:
        # electrons
        ax = axs[i]
        axs[i].set_prop_cycle('color', plt.cm.Greens_r(np.linspace(0,1, len(meta_e['channels_dict_df'])+color_offset)))
        if wind:
            for ch in np.arange(1, len(meta_e['channels_dict_df'])):
                ax.plot(edic.index, edic[f'FLUX_{ch}'] * wind_ev2MeV_fac, label='Wind/3DP '+meta_e['channels_dict_df']['Bins_Text'].values[ch], drawstyle='steps-mid')
        
        
        if plot_ephin:
            ax.plot(ephin.index, ephin[eph_e_ch]*intercal, '-k', label='SOHO/EPHIN '+meta_ephin[eph_e_ch]+f' / {intercal}', drawstyle='steps-mid')
        # ax.set_ylim(1e0, 1e4)
        ax.legend(title='Electrons', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONT_LEGEND)
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label, fontsize=FONT_YLABEL)
        i += 1
        
    color_offset = 2    
    if plot_protons:    
        # protons low en:
        ax = axs[i]
        ax.set_prop_cycle('color', plt.cm.plasma(np.linspace(0,1, len(meta_p['channels_dict_df'])+color_offset)))
        if wind:
            for ch in np.arange(2, len(meta_p['channels_dict_df'])):
                ax.plot(pdic.index, pdic[f'FLUX_{ch}'] * wind_ev2MeV_fac, label='Wind/3DP '+meta_p['channels_dict_df']['Bins_Text'].values[ch],
                        drawstyle='steps-mid')
        ax.legend(title='Protons', loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label, fontsize=FONT_YLABEL)

        # protons high en:
        if erne:
            ax.set_prop_cycle('color', plt.cm.YlOrRd(np.linspace(0.2,1,10))) #cm.RdPu_r
            for ch in np.arange(0, 10):
                ax.plot(erne_p.index, erne_p[f'PH_{ch}'], label='SOHO/ERNE/HED '+meta_erne['channels_dict_df_p']['ch_strings'][ch], 
                            drawstyle='steps-mid')
        ax.legend(title='Protons', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONT_LEGEND)
        ax.set_yscale('log')
        i += 1

    
    if plot_mag:    
        ax = axs[i]
        ax.plot(df_mag.index, df_mag.B.values, label='B', color='k', linewidth=1)
        ax.plot(df_mag.index, df_mag.BRTN_0.values, label='Br', color='dodgerblue', linewidth=1)
        ax.plot(df_mag.index, df_mag.BRTN_1.values, label='Bt', color='limegreen', linewidth=1)
        ax.plot(df_mag.index, df_mag.BRTN_2.values, label='Bn', color='deeppink', linewidth=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))#, title='RTN')#, bbox_to_anchor=(1, 0.5))
        ax.set_ylabel('B [nT]', fontsize=FONT_YLABEL)
        ax.tick_params(axis="x",direction="in", which='both') #, pad=-15
        i += 1
        
    if plot_polarity:
        pos = get_horizons_coord('Wind', time={'start':df_mag_pol.index[0]-pd.Timedelta(minutes=15),
                                            'stop':df_mag_pol.index[-1]+pd.Timedelta(minutes=15),'step':"1min"}) 
                                                # (lon, lat, radius) in (deg, deg, AU)
        pos = pos.transform_to(frames.HeliographicStonyhurst())
        #Interpolate position data to magnetic field data cadence
        r = np.interp([t.timestamp() for t in df_mag_pol.index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.radius.value)
        lat = np.interp([t.timestamp() for t in df_mag_pol.index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.lat.value)
        pol, phi_relative = polarity_rtn(df_mag_pol.BRTN_0.values, df_mag_pol.BRTN_1.values, df_mag_pol.BRTN_2.values,r,lat,V=400)
    # create an inset axe in the current axe:
        pol_ax = inset_axes(ax, height="5%", width="100%", loc=9, bbox_to_anchor=(0.,0,1,1.1), bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
        pol_ax.get_xaxis().set_visible(False)
        pol_ax.get_yaxis().set_visible(False)
        pol_ax.set_ylim(0,1)
        pol_ax.set_xlim([df_mag_pol.index.values[0], df_mag_pol.index.values[-1]])
        pol_arr = np.zeros(len(pol))+1
        timestamp = df_mag_pol.index.values[2] - df_mag_pol.index.values[1]
        norm = Normalize(vmin=0, vmax=180, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        pol_ax.bar(df_mag_pol.index.values[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
        pol_ax.bar(df_mag_pol.index.values[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
        pol_ax.set_xlim(startdate, enddate)
        
        
    if plot_mag_angles:
        alpha, phi = mag_angles(df_mag.B.values, df_mag.BRTN_0.values, df_mag.BRTN_1.values, df_mag.BRTN_2.values)
        ax = axs[i]
        ax.plot(df_mag.index, alpha, '.k', label='alpha', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-90, 90)
        ax.set_ylabel(r"$\Theta_\mathrm{B}$ [°]", fontsize=FONT_YLABEL)
        ax.tick_params(axis="x",direction="in")
        i += 1
        
        ax = axs[i]
        ax.plot(df_mag.index, phi, '.k', label='phi', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-180, 180)
        ax.set_ylabel(r"$\Phi_\mathrm{B}$ [°]", fontsize=FONT_YLABEL)
        ax.tick_params(axis="x",direction="in", which='both')
        i += 1
        
    ### Temperature
    if plot_T:
        axs[i].plot(df_vsw.index, df_vsw['ion_temp'], '-k', label="Temperature")
        axs[i].set_ylabel(r"T$_\mathrm{p}$ [K]", fontsize=FONT_YLABEL)
        i += 1

    ### Density
    if plot_N:
        axs[i].plot(df_vsw.index, df_vsw.ion_density,
                    '-k', label="Ion density")
        axs[i].set_ylabel(r"N$_\mathrm{p}$ [cm$^{-3}$]", fontsize=FONT_YLABEL)
        i += 1

    ### Sws
    if plot_Vsw:
        axs[i].plot(df_vsw.index, df_vsw.vsw,
                    '-k', label="Bulk speed")
        axs[i].set_ylabel(r"V$_\mathrm{sw}$ [km/s]", fontsize=FONT_YLABEL)
        i += 1
            

    axs[0].set_title('Near-Earth spacecraft (Wind, SOHO)', fontsize=FONT_YLABEL)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%b %d'))
    axs[-1].xaxis.set_tick_params(rotation=0)
    axs[-1].set_xlabel(f"Time (UTC) / Date in {startdate.year}", fontsize=15)
    axs[-1].set_xlim(startdate, enddate)
    fig.patch.set_facecolor('white')
    fig.set_dpi(200)
    plt.show()
    # if save_fig:
    #     plt.savefig(f'{outpath}L1_multiplot_{str(startdate.date())}--{str(enddate.date())}_{av_sep}.png')