import os
import sunpy

from sunpy.net import Fido
from sunpy.net import attrs as a

from sunpy.net import Scraper
from sunpy.time import TimeRange
from sunpy.data.data_manager.downloader import ParfiveDownloader

def cdaweb_download_fido(dataset, startdate, enddate, path=None, max_conn=5):
    """
    Downloads dataset files via SunPy/Fido from CDAWeb

    Parameters
    ----------
    dataset : {str}
        Name of dataset:
        - 'PSP_FLD_L3_RFS_HFR'
        - 'PSP_FLD_L3_RFS_LFR'
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or
        "standard" datetime string (e.g., "2021/04/15") (enddate must always be
        later than startdate)
    path : {str}, optional
        Local path for storing downloaded data, by default None
    max_conn : {int}, optional
        The number of parallel download slots used by Fido.fetch, by default 5

    Returns
    -------
    List of downloaded files
    """
    trange = a.Time(startdate, enddate)
    cda_dataset = a.cdaweb.Dataset(dataset)
    try:
        result = Fido.search(trange, cda_dataset)
        filelist = [i[0].split('/')[-1] for i in result.show('URL')[0]]
        filelist.sort()
        if path is None:
            filelist = [sunpy.config.get('downloads', 'download_dir') + os.sep + file for file in filelist]
        elif type(path) is str:
            filelist = [path + os.sep + f for f in filelist]
        downloaded_files = filelist

        # Check if file with same name already exists in path
        for i, f in enumerate(filelist):
            if os.path.exists(f) and os.path.getsize(f) == 0:
                os.remove(f)
            if not os.path.exists(f):
                downloaded_file = Fido.fetch(result[0][i], path=path, max_conn=max_conn)
    except (RuntimeError, IndexError):
        print(f'Unable to obtain "{dataset}" data for {startdate}-{enddate}!')
        downloaded_files = []
    return downloaded_files

# make this generally for any instrument? 
def download_wind_waves_cdf(sensor, startdate, enddate, path=None):
    """
    Download a single Wind WAVES file (L2 only) with ParfiveDownloader class.

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

        scrap = Scraper(pattern=pattern, sensor=sensor.lower(), version=".*")   # handles any version

        # for some reason includes the day preceding startdate, so just remove it
        # also includes the enddate as well, remove that to keep the scheme consistent
        filelist_urls = scrap.filelist(timerange=timerange)[1:-1]

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
        # in case of error, should probably clear the directory of any successful downloads? i dunno

    return downloaded_files

if __name__ == "__main__":
    files = download_wind_waves_cdf("RAD1", "2024/04/19", "2024/04/21")