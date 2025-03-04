import astropy.units as u
import datetime as dt
# import math
import numpy as np
import os
import pandas as pd
import warnings

from astropy.constants import e, k_B, m_p
from astropy.table import QTable
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from seppy.loader.psp import calc_av_en_flux_PSP_EPIHI, psp_isois_load
from seppy.tools import resample_df
from stixdcpy.quicklook import LightCurves # https://github.com/i4Ds/stixdcpy
from sunpy.coordinates import frames, get_horizons_coord

from my_func_py3 import mag_angles
from polarity_plotting import polarity_rtn

# disable unused speasy data provider before importing to speed it up
os.environ['SPEASY_CORE_DISABLED_PROVIDERS'] = "sscweb,archive,csa"
import speasy as spz

# omit Pandas' PerformanceWarning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('axes', titlesize=20)  # fontsize of the axes title
plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
plt.rcParams['agg.path.chunksize'] = 20000

import ipywidgets as w
import seppy.tools.widgets as seppyw

class GUI():
    def __init__(self):
        self._options = ["RFS", "STIX", "EPI-lo electrons", "EPI-hi electrons", "EPI-lo protons", "EPI-hi protons", "MAG", "MAG angles", "V_sw", "N", "T", "P_dyn", "Polarity"]
        self._boxes = dict(zip([w.Checkbox(value=False, description=quant, indent=False) for quant in self._options], self._options))
        for option in self._boxes.keys():
            display(self._boxes[option])
    
    

def load_data(opt):
    """
    Load data used in plotting.
    
    Parameters
    ----------

    options : {dict} 
        dictionary of plotting options

    Returns
    -------

    list of dataframes

    """
    

    data = {}

    #####################################################################
    ######## Data loading ###############################################
    #####################################################################

    if opt["plot_epilo_e"] or opt["plot_epihi_e"]:
        opt["plot_electrons"] = True
    else:
        opt["plot_electrons"] = False

    if opt["plot_epilo_p"] or opt["plot_epihi_p"]:
        opt["plot_protons"] = True
    else:
        opt["plot_protons"] = False

    if opt["plot_stix"]:
        if opt["enddate"]-opt["startdate"] > dt.timedelta(days=7):
            print('WARNING: STIX loading for more than 7 days not supported at the moment!')
            print('')
        lc = LightCurves.from_sdc(start_utc=opt["startdate"], end_utc=opt["enddate"], ltc=opt["stix_ltc"])
        data["stix_orig"] = lc.to_pandas()
        
    if opt["plot_epihi_p"] or opt["plot_epihi_e"]:
        data["psp_het_org"], data["psp_het_energies"] = psp_isois_load('PSP_ISOIS-EPIHI_L2-HET-RATES60', opt["startdate"], opt["enddate"], 
                                                                       path=opt["file_path"], resample=None)

    if opt["plot_epilo_e"]:
        data["psp_epilo_org"], data["psp_epilo_energies_org"] = psp_isois_load('PSP_ISOIS-EPILO_L2-PE', opt["startdate"], opt["enddate"], 
                                                                               path=opt["file_path"], resample=None, epilo_channel=opt["epilo_channel"], 
                                                                               epilo_threshold=None)
        electron_countrate_keys = data["psp_epilo_org"].filter(like='Electron_CountRate_ChanF_E').keys()
        data["psp_epilo_org"][electron_countrate_keys] = data["psp_epilo_org"][electron_countrate_keys].mask(data["psp_epilo_org"][electron_countrate_keys] < 0.0)
        

    if opt["plot_epilo_p"]:
        data["psp_epilo_ic_org"], data["psp_epilo_ic_energies_org"] = psp_isois_load('PSP_ISOIS-EPILO_L2-IC', opt["startdate"], opt["enddate"], 
                                                                                     path=opt["file_path"], resample=None, epilo_channel=opt["epilo_ic_channel"], 
                                                                                     epilo_threshold=None)

    if opt["plot_rfs"]:
        psp_rfs_lfr_psd = spz.get_data(spz.inventories.data_tree.cda.ParkerSolarProbe.PSP_FLD.RFS_LFR.PSP_FLD_L3_RFS_LFR.psp_fld_l3_rfs_lfr_PSD_SFU,
                                        opt["startdate"], opt["enddate"]).replace_fillval_by_nan()
        psp_rfs_hfr_psd = spz.get_data(spz.inventories.data_tree.cda.ParkerSolarProbe.PSP_FLD.RFS_HFR.PSP_FLD_L3_RFS_HFR.psp_fld_l3_rfs_hfr_PSD_SFU, 
                                        opt["startdate"], opt["enddate"]).replace_fillval_by_nan()

        # Get frequency bins, since metadata is lost upon conversion to df
        psp_rfs_lfr_freq = psp_rfs_lfr_psd.axes[1].values[0] / 1e6  # in MHz
        psp_rfs_hfr_freq = psp_rfs_hfr_psd.axes[1].values[0] / 1e6

        df_psp_rfs_lfr_psd_o = psp_rfs_lfr_psd.to_dataframe()
        df_psp_rfs_hfr_psd_o = psp_rfs_hfr_psd.to_dataframe()

        # put frequencies into column names for easier access
        df_psp_rfs_lfr_psd_o.columns = psp_rfs_lfr_freq
        df_psp_rfs_hfr_psd_o.columns = psp_rfs_hfr_freq

    if opt["plot_mag"]:
        df_psp_mag_rtn = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.FIELDS_MAG.psp_mag_1min.psp_b_1min, 
                                    opt["startdate"], opt["enddate"], output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_mag_phi = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.FIELDS_MAG.psp_mag_1min.psp_b_1min_phi, 
                                    opt["startdate"], opt["enddate"], output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_mag_theta = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.FIELDS_MAG.psp_mag_1min.psp_b_1min_theta, 
                                    opt["startdate"], opt["enddate"], output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_mag_tot = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.FIELDS_MAG.psp_mag_1min.psp_b_1min_tot, 
                                    opt["startdate"], opt["enddate"], output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        
        data["psp_mag"] = pd.concat([df_psp_mag_rtn, df_psp_mag_phi, df_psp_mag_theta, df_psp_mag_tot], axis=1)
        data["psp_mag"]['phi_mod'] = ((data["psp_mag"]['phi'].values - 180) % 360) - 180 

    if opt["plot_mag_angles"]:
        theta, phi = mag_angles(data["psp_mag"]['|b|'].values, data["psp_mag"]['br'].values, data["psp_mag"]['bt'].values, data["psp_mag"]['bn'].values)
        data["psp_mag"]['theta2'] = theta
        data["psp_mag"]['phi2'] = phi



    #################################################################
    ############## Resampling #######################################
    #################################################################

    if opt["resample"] is not None:
        if opt["plot_epihi_e"] or opt["plot_epihi_p"]:
            data["psp_het"] = resample_df(data["psp_het_org"], opt["resample"])   
        if opt["plot_epilo_e"]:
            data["psp_epilo"] = resample_df(data["psp_epilo_org"], opt["resample"]) 
        if opt["plot_epilo_p"]:
            data["psp_epilo_ic"] = resample_df(data["psp_epilo_ic_org"], opt["resample"]) 
        # if plot_psp_pixel:
        #     df_psp_pixel = resample_df(df_psp_pixel_org, resample) 
        if opt["plot_Vsw"] or opt["plot_N"] or opt["plot_T"]:
            df_magplas_spani = resample_df(df_psp_spani, opt["resample_mag"]) 
            df_magplas_spc = resample_df(df_psp_spc, opt["resample_mag"])
        if opt["plot_mag"]:
            data["mag"] = resample_df(data["psp_mag"], opt["resample_mag"]) 
        if opt["plot_stix"]:
            data["stix"] = resample_df(data["stix_orig"], opt["resample"])
        if opt["plot_rfs"]:
            data["psp_rfs_hfr_psd"] = resample_df(df_psp_rfs_hfr_psd_o, opt["resample"], origin="start_day")
            data["psp_rfs_lfr_psd"] = resample_df(df_psp_rfs_lfr_psd_o, opt["resample"], origin="start_day")
        
        else:
            if opt["plot_epihi_e"] or opt["plot_epihi_p"]:
                data["psp_het"] = data["psp_het_org"] 
            if opt["plot_epilo_e"]:
                data["psp_epilo"] = data["psp_epilo_org"]
            if opt["plot_epilo_p"]:
                data["psp_epilo_ic"] = data["psp_epilo_ic_org"]
            # if plot_psp_pixel:
            #     df_psp_pixel = df_psp_pixel_org
            if opt["plot_Vsw"] or opt["plot_N"] or opt["plot_T"]:
                df_magplas_spani = df_psp_spani 
                df_magplas_spc = df_psp_spc
            if opt["plot_mag"]:
                data["mag"] = data["psp_mag"]
            if opt["plot_stix"]:
                data["stix"] = data["stix_orig"]
            if opt["plot_rfs"]:
                data["psp_rfs_hfr_psd"] = df_psp_rfs_hfr_psd_o
                data["psp_rfs_lfr_psd"] = df_psp_rfs_lfr_psd_o
                # Remove bar artifacts caused by non-NaN values before time jumps
                for i in range(len(data["psp_rfs_lfr_psd"].index) - 1):
                    if (data["psp_rfs_lfr_psd"].index[i+1] - data["psp_rfs_lfr_psd"].index[i]) > np.timedelta64(5, "m"):   
                        data["psp_rfs_lfr_psd"].iloc[i,:] = np.nan
                for i in range(len(data["psp_rfs_hfr_psd"].index) - 1):
                    if (data["psp_rfs_hfr_psd"].index[i+1] - data["psp_rfs_hfr_psd"].index[i]) > np.timedelta64(5, "m"):
                        data["psp_rfs_hfr_psd"].iloc[i,:] = np.nan



    ############################################################################
    ############## Energy channel ranges #######################################
    ############################################################################

    if opt["plot_protons"]:  
        #Channels list
        
        opt["channels_n_psp_het_p"] = list(np.arange(0, len(data["psp_het"].filter(like=f'{opt["psp_het_viewing"]}_H_Flux_').keys()), opt["n_psp_het_p"]))
        opt["channels_n_psp_epilo_ic"] = list(np.arange(0, 31, opt["n_psp_epilo_ic"]))

        #Chosen channels
        print('Chosen proton channels:')
        print('psp_het_p:', opt["channels_n_psp_het_p"], ',', len(opt["channels_n_psp_het_p"]))
        print('psp_epilo_ic:', opt["channels_n_psp_epilo_ic"], ',', len(opt["channels_n_psp_epilo_ic"]))

    if opt["plot_electrons"]:
        opt["channels_n_psp_het_e"] = list(np.arange(0, len(data["psp_het"].filter(like=f'{opt["psp_het_viewing"]}_Electrons_Rate_').keys()), opt["n_psp_het_e"]))
        opt["channels_n_psp_epilo_e"] = list(np.arange(3, 8, opt["n_psp_epilo_e"])) # list(np.arange(0, len(data["psp_epilo"].filter(like='Electron_CountRate_ChanF').keys()), n_psp_epilo_e))

        print('Chosen electron channels:')
        print('psp_epilo_e:', opt["channels_n_psp_epilo_e"], ',', len(opt["channels_n_psp_epilo_e"]))
        print('psp_het_e:', opt["channels_n_psp_het_e"], ',', len(opt["channels_n_psp_het_e"]))

    return data

def make_plot(data, opt):
    """
    Plot chosen data with user-specified parameters.
    """

    panels = 1*opt["plot_rfs"] + 1*opt["plot_stix"] + 1*opt["plot_electrons"] + 1*opt["plot_protons"] + 2*opt["plot_mag_angles"] + 1*opt["plot_mag"] #+ 1*plot_Vsw + 1*plot_N + 1*plot_T + 1*plot_p_dyn 

    panel_ratios = list(np.zeros(panels)+1)

    if opt["plot_rfs"]:
        panel_ratios[0] = 2

    if opt["plot_electrons"] and opt["plot_protons"]:
        panel_ratios[0+1*opt["plot_stix"]+1*opt["plot_rfs"]] = 2
        panel_ratios[1+1*opt["plot_stix"]+1*opt["plot_rfs"]] = 2
    if opt["plot_electrons"] or opt["plot_protons"]:    
        panel_ratios[0+1*opt["plot_stix"]+1*opt["plot_rfs"]] = 2

    FONT_YLABEL = 20
    FONT_LEGEND = 10
    
    if panels == 3:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 4*panels])#, gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")
    else:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 3*panels], gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")

    fig.subplots_adjust(hspace=0.1)
    
    i = 0

    if opt["plot_rfs"]:
        vmin, vmax = 500, 1e7
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        
        TimeHFR2D, FreqHFR2D = np.meshgrid(data["psp_rfs_hfr_psd"].index, data["psp_rfs_hfr_psd"].columns, indexing='ij')
        TimeLFR2D, FreqLFR2D = np.meshgrid(data["psp_rfs_lfr_psd"].index, data["psp_rfs_lfr_psd"].columns, indexing='ij')

        # Create colormeshes. Shading option flat and thus the removal of last row and column are there to solve the time jump bar problem, 
        # when resampling isn't used
        mesh = axs[i].pcolormesh(TimeLFR2D, FreqLFR2D, data["psp_rfs_lfr_psd"].iloc[:-1,:-1], shading='flat', cmap='jet', norm=log_norm)
        axs[i].pcolormesh(TimeHFR2D, FreqHFR2D, data["psp_rfs_hfr_psd"].iloc[:-1,:-1], shading='flat', cmap='jet', norm=log_norm)

        axs[i].set_yscale('log')
        axs[i].set_ylabel("Frequency (MHz)", fontsize=FONT_YLABEL)
        
        # Add inset axes for colorbar
        axins = inset_axes(axs[i], width="100%", height="100%", loc="center", bbox_to_anchor=(1.05,0,0.03,1), bbox_transform=axs[i].transAxes, borderpad=0.2)
        cbar = fig.colorbar(mesh, cax=axins, orientation="vertical")
        cbar.set_label("Intensity (sfu)", rotation=90, labelpad=10, fontsize=FONT_YLABEL)
        i += 1
        
    
    if opt["plot_stix"]:
        for key in data["stix"].keys():
            axs[i].plot(data["stix"].index, data["stix"][key], ds="steps-mid", label=key)
        if opt["stix_ltc"]:
            title = 'SolO/STIX (light travel time corrected)'
        else:
            title = 'SolO/STIX'
        if opt["legends_inside"]:
            axs[i].legend(loc='upper right', title=title)
        else:
            # axs[i].legend(loc='upper right', title=title, bbox_to_anchor=(1, 0.5))
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', title=title)
        axs[i].set_ylabel('Counts', fontsize=FONT_YLABEL)
        axs[i].set_yscale('log')
        i +=1  
    
    
    color_offset = 4 
    
    if opt["plot_electrons"]:
        if opt["plot_epilo_e"]:
            axs[i].set_prop_cycle('color', plt.cm.viridis_r(np.linspace(0, 1, len(opt["channels_n_psp_epilo_e"])+color_offset)))
            for channel in opt["channels_n_psp_epilo_e"]:
                psp_epilo_energy = np.round(data["psp_epilo_energies_org"][f'Electron_Chan{opt["epilo_channel"]}_Energy'][f'Electron_Chan{opt["epilo_channel"]}_Energy_E{channel}_P{opt["epilo_viewing"]}'], 2).astype(str)
                axs[i].plot(data["psp_epilo"].index, data["psp_epilo"][f'Electron_CountRate_Chan{opt["epilo_channel"]}_E{channel}_P{opt["epilo_viewing"]}'],
                            ds="steps-mid", label=f'EPI-lo PE {opt["epilo_channel"]}{opt["epilo_viewing"]} {psp_epilo_energy} keV')
    
        if opt["plot_epihi_e"]:
            axs[i].set_prop_cycle('color', plt.cm.Reds_r(np.linspace(0, 1, len(opt["channels_n_psp_het_e"])+color_offset)))
            for channel in opt["channels_n_psp_het_e"]:
                axs[i].plot(data["psp_het"].index, data["psp_het"][f'{opt["psp_het_viewing"]}_Electrons_Rate_{channel}'],
                            ds="steps-mid", label=f'HET {opt["psp_het_viewing"]}'+data["psp_het_energies"]['Electrons_ENERGY_LABL'].flatten()[channel])
                
        # axs[i].set_ylabel("Flux\n"+r"[(cm$^2$ sr s MeV)$^{-1}]$", fontsize=FONT_YLABEL)
        axs[i].set_ylabel("Count rates", fontsize=FONT_YLABEL)
        if opt["legends_inside"]:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                          title=f'Electrons',
                          fontsize=FONT_LEGEND)
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., 
                          title=f'Electrons',
                          fontsize=FONT_LEGEND)
       
        axs[i].set_yscale('log')
        i +=1    
    
        
    color_offset = 2    
    if opt["plot_protons"]:
        if opt["plot_epilo_p"]:
            axs[i].set_prop_cycle('color', plt.cm.viridis_r(np.linspace(0, 1, len(opt["channels_n_psp_epilo_ic"])+color_offset)))
            # [::-1] to reverse list
            for channel in opt["channels_n_psp_epilo_ic"][::-1]:
                # print(f'H_Flux_Chan{epilo_ic_channel}_E{channel}_P{epilo_ic_viewing}')
                psp_epilo_ic_energy = np.round(data["psp_epilo_ic_energies_org"][f'H_Chan{opt["epilo_ic_channel"]}_Energy'][f'H_Chan{opt["epilo_ic_channel"]}_Energy_E{channel}_P{opt["epilo_ic_viewing"]}'], 2).astype(str)
                axs[i].plot(data["psp_epilo_ic"].index, data["psp_epilo_ic"][f'H_Flux_Chan{opt["epilo_ic_channel"]}_E{channel}_P{opt["epilo_ic_viewing"]}'],
                            ds="steps-mid", label=f'EPI-lo IC {opt["epilo_ic_channel"]}{opt["epilo_ic_viewing"]} {psp_epilo_ic_energy} keV')
    
        # if plot_psp_pixel:
        #     axs[i].set_prop_cycle('color', plt.cm.tab10(range(6)))
        #     for key in ['L2Ap', 'L3Ap', 'L4Ap', 'H2Ap', 'H3Ap', 'H4Ap']:
        #     # for key in ['L2Ap', 'L4Ap', 'H2Ap', 'H3Ap', 'H4Ap']:
        #         axs[i].plot(df_psp_pixel.index, df_psp_pixel[f'{key}_Flux'], label=f'{key} {energies_psp_pixel[key]}', drawstyle='steps-mid')
        
        if opt["plot_epihi_p"]:    
            if opt["plot_epihi_p_combined_pixels"]:
                # comb_channels = [[1,2], [3,5], [5,7], [4,5], [7], [9]]
                comb_channels = [[3,5], [5,7], [4,5], [7], [9]]
                axs[i].set_prop_cycle('color', plt.cm.Greys_r(np.linspace(0, 1, len(comb_channels)+5)))
                for channel in comb_channels:
                    df_psp_epihi, df_psp_epihi_name = calc_av_en_flux_PSP_EPIHI(data["psp_het"], data["psp_het_energies"], channel, 'p', 'het', opt["psp_het_viewing"])
                    axs[i].plot(df_psp_epihi.index, df_psp_epihi.flux, label=f'HET {opt["psp_het_viewing"]}{df_psp_epihi_name}', lw=1, ds="steps-mid")
            else:
                axs[i].set_prop_cycle('color', plt.cm.plasma(np.linspace(0, 1, len(opt["channels_n_psp_het_p"])+color_offset)))
                for channel in opt["channels_n_psp_het_p"]:
                    axs[i].plot(data["psp_het"].index, data["psp_het"][f'{opt["psp_het_viewing"]}_H_Flux_{channel}'], label=f'HET {opt["psp_het_viewing"]}'+data["psp_het_energies"]['H_ENERGY_LABL'].flatten()[channel], ds="steps-mid")
        
        axs[i].set_ylabel("Flux\n"+r"[(cm$^2$ sr s MeV)$^{-1}]$", fontsize=FONT_YLABEL)
        # title = f'Ions (HET {psp_het_viewing})'
        title = f'Ions (Pixel)'
        if opt["legends_inside"]:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                          title=title,
                          fontsize=FONT_LEGEND)
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., 
                          title=title,
                          fontsize=FONT_LEGEND)
        axs[i].set_yscale('log')
    
        # axs[i].set_ylim([5e-2, None])
        
        i +=1    
        
        
    # plot magnetic field
    if opt["plot_mag"]:
        ax = axs[i]
        ax.plot(data["mag"].index, data["mag"]['|b|'], label='B', color='k', linewidth=1)
        ax.plot(data["mag"].index.values, data["mag"]['br'].values, label='Br', color='dodgerblue')
        ax.plot(data["mag"].index.values, data["mag"]['bt'].values, label='Bt', color='limegreen')
        ax.plot(data["mag"].index.values, data["mag"]['bn'].values, label='Bn', color='deeppink')
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        if opt["legends_inside"]:
            ax.legend(loc='upper right')
        else:
            # ax.legend(loc='upper right', bbox_to_anchor=(1.01, 0.5))
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            
        ax.set_ylabel('B [nT]', fontsize=FONT_YLABEL)
        ax.tick_params(axis="x", direction="in", which='both')#, pad=-15)
        i += 1
        
    if opt["plot_polarity"]:
        pos = get_horizons_coord(f'PSP', time={'start':data["mag"].index[0]-pd.Timedelta(minutes=15), 'stop':data["mag"].index[-1]+pd.Timedelta(minutes=15), 'step':"1min"})  # (lon, lat, radius) in (deg, deg, AU)
        pos = pos.transform_to(frames.HeliographicStonyhurst())
        #Interpolate position data to magnetic field data cadence
        r = np.interp([t.timestamp() for t in data["mag"].index], [t.timestamp() for t in pd.to_datetime(pos.obstime.value)], pos.radius.value)
        lat = np.interp([t.timestamp() for t in data["mag"].index], [t.timestamp() for t in pd.to_datetime(pos.obstime.value)], pos.lat.value)
        pol, phi_relative = polarity_rtn(data["mag"]['br'].values, data["mag"]['bt'].values, data["mag"]['bn'].values, r, lat, V=400)
        # create an inset axe in the current axe:
        pol_ax = inset_axes(ax, height="5%", width="100%", loc='upper center', bbox_to_anchor=(0.,0,1,1.1), bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
        pol_ax.get_xaxis().set_visible(False)
        pol_ax.get_yaxis().set_visible(False)
        pol_ax.set_ylim(0,1)
        pol_ax.set_xlim([data["mag"].index.values[0], data["mag"].index.values[-1]])
        pol_arr = np.zeros(len(pol))+1
        timestamp = data["mag"].index.values[2] - data["mag"].index.values[1]
        norm = Normalize(vmin=0, vmax=180, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        pol_ax.bar(data["mag"].index.values[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
        pol_ax.bar(data["mag"].index.values[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
        pol_ax.set_xlim(opt["startdate"], opt["enddate"])
        
    if opt["plot_mag_angles"]:
        ax = axs[i]
        #Bmag = np.sqrt(np.nansum((mag_data.B_r.values**2,mag_data.B_t.values**2,mag_data.B_n.values**2), axis=0))    
        # alpha, phi = mag_angles(data["mag"].BFIELD_3, data["mag"].BFIELD_0.values, data["mag"].BFIELD_1.values,
        #                         data["mag"].BFIELD_2.values)
        ax.plot(data["mag"].index, data["mag"]['theta'], '.k', label='theta', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-90, 90)
        ax.set_ylabel(r"$\Theta_\mathrm{B}$ [°]", fontsize=FONT_YLABEL)
        ax.tick_params(axis="x",direction="in", pad=-15)
    
        i += 1
        ax = axs[i]
        # ax.plot(data["mag"].index, data["mag"]['phi'], '.k', label='phi', ms=1)
        ax.plot(data["mag"].index, data["mag"]['phi_mod'], '.k', label='phi', ms=1)
        # ax.plot(data["mag"].index, data["mag"]['phi2'], '.r', label='phi', ms=1)    
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-180, 180)
        ax.set_ylabel(r"$\Phi_\mathrm{B}$ [°]", fontsize=FONT_YLABEL)
        ax.tick_params(axis="x",direction="in", which='both', pad=-15)
        i += 1
        
    ### Temperature
    if opt["plot_T"]:
        axs[i].plot(df_magplas_spani.index, df_magplas_spani['T_K'], '-k', label="SPAN-i")
        axs[i].plot(df_magplas_spc.index, df_magplas_spc['T'], '-r', label="SPC")
        axs[i].set_ylabel(r"T$_\mathrm{p}$ [K]", fontsize=FONT_YLABEL)
        axs[i].set_yscale('log')
    
        # TODO: manually set lower boundary, remove at some point
        axs[i].set_ylim(np.nanmin(df_magplas_spc['T'])-0.1*np.nanmin(df_magplas_spc['T']), None)
    
        if opt["legends_inside"]:
            axs[i].legend(loc='upper right')
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        i += 1
    
    ### Dynamic pressure
    if opt["plot_p_dyn"]:
        axs[i].plot(df_magplas_spani.index, df_magplas_spani['p_dyn'], '-k', label="SPAN-i")
        axs[i].plot(df_magplas_spc.index, df_magplas_spc['p_dyn'], '-r', label="SPC")
        axs[i].set_ylabel(r"P$_\mathrm{dyn}$ [nPa]", fontsize=FONT_YLABEL)
        if opt["legends_inside"]:
            axs[i].legend(loc='upper right')
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        axs[i].set_yscale('log')
        i += 1
    
    ### Density
    if opt["plot_N"]:
        axs[i].plot(df_magplas_spani.index, df_magplas_spani['Density'], '-k', label="SPAN-i")
        axs[i].plot(df_magplas_spc.index, df_magplas_spc['np_tot'], '-r', label="SPC")
        axs[i].set_ylabel(r"N$_\mathrm{p}$ [cm$^{-3}$]", fontsize=FONT_YLABEL)
        if opt["legends_inside"]:
            axs[i].legend(loc='upper right')
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        axs[i].set_yscale('log')
        i += 1
    
    ### Vsw
    if opt["plot_Vsw"]:
        axs[i].plot(df_magplas_spani.index, df_magplas_spani['V_tot_rtn'], '-k', label="SPAN-i")
        axs[i].plot(df_magplas_spc.index, df_magplas_spc['|vp_tot|'], '-r', label="SPC")
        axs[i].set_ylabel(r"V$_\mathrm{sw}$ [kms$^{-1}$]", fontsize=FONT_YLABEL)
        if opt["legends_inside"]:
            axs[i].legend(loc='upper right')
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        # i += 1     
            
    #axs[-1].set_xlabel(f"Date in {year}/  Time (UTC)", fontsize=15)
    #axs[-1].set_xlim(startdate, enddate)
    axs[0].set_title(f'Parker Solar Probe', ha='center')
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
    axs[-1].xaxis.set_tick_params(rotation=0)
    axs[-1].set_xlabel(f"Time (UTC) / Date in {opt["year"]}")#, fontsize=15)
    axs[-1].set_xlim(opt["startdate"], opt["enddate"])
    
    #plt.tight_layout()
    fig.patch.set_facecolor('white')
    fig.set_dpi(200)
    plt.show()


    