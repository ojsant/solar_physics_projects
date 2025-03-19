import datetime as dt
import ipywidgets as w

import tools.stereo_tools_alt_with_gui as stereo
import tools.psp_tools_alt as psp
import tools.l1_tools_alt as l1
import tools.solo_tools as solo


style = {'description_width' : '50%'} 

common_attrs = ["spacecraft", "startdate", "enddate", "resample", "resample_mag", "resample_pol", "pos_timestamp", "radio_cmap", "legends_inside"]

variable_attrs = ['radio', 'mag', 'mag_angles', 'polarity', 'Vsw', 'N', 'T', 'p_dyn', 'pad']

psp_attrs = ['epilo_e', 'epilo_p', 'epihi_e', 'epihi_p', 'het_viewing', 'epilo_viewing', 'epilo_ic_viewing', 'epilo_channel', 'epilo_ic_channel']

stereo_attrs = ['sc', 'sept_e', 'sept_p', 'het_e', 'het_p', 'sept_viewing', 'ch_sept_p', 'ch_sept_e', 'ch_het_p', 'ch_het_e']

l1_attrs = ['wind_e', 'wind_p', 'ephin', 'erne', 'ch_eph_e', 'ch_eph_p', 'intercal']

solo_attrs = ['stix']

class Options:
    def __init__(self):
        # now = dt.datetime.now()
        # dt_now = dt.datetime(now.year, now.month, now.day, 0, 0)
        self._sc_attrs = None
        self.spacecraft = w.Dropdown(description="Spacecraft", options=["PSP", "SolO", "L1 (Wind/SOHO)", "STEREO"], style=style)
        self.startdate = w.NaiveDatetimePicker(value=dt.datetime(2022, 3, 14), disabled=False, description="Start date:")
        self.enddate = w.NaiveDatetimePicker(value=dt.datetime(2022, 3, 16), disabled=False, description="End date:")

        self.resample = w.BoundedIntText(value=15, min=0, max=30, step=1, description='Resampling (min):', disabled=False, style=style)
        self.resample_mag = w.BoundedIntText(value=5, min=0, max=30, step=1, description='MAG resampling (min):', disabled=False, style=style)
        self.resample_pol = w.BoundedIntText(value=1, min=0, max=30, step=1, description='Polarity resampling (min):', disabled=False, style=style)
        self.radio_cmap = w.Dropdown(options=['jet', 'magma', 'Spectral'], value='jet', description='Radio colormap', style=style)
        self.pos_timestamp = w.Dropdown(options=['center', 'start', 'original'], description='Timestamp position', style=style)
        self.legends_inside = w.Checkbox(value=False, description='Legends inside')

        self.radio = w.Checkbox(value=True, description="Radio")
        self.pad = w.Checkbox(value=False, description="Pitch angle distribution", disabled=True)    # TODO: remove disabled keyword after implementation
        self.mag = w.Checkbox(value=True, description="MAG")
        self.mag_angles = w.Checkbox(value=True, description="MAG angles")
        self.polarity = w.Checkbox(value=True, description="Polarity")
        self.Vsw = w.Checkbox(value=True, description="V_sw")
        self.N = w.Checkbox(value=True, description="N")
        self.T = w.Checkbox(value=True, description="T")
        self.p_dyn = w.Checkbox(value=True, description="p_dyn")
        
        self.path = None
        self.plot_range = date_selector(self.startdate.value, self.enddate.value)

        
        ########### Define widget layout ###########
        
        def change_sc(change):
            """
            Change dynamic attributes of current Options object and update spacecraft-specific widget display.

            When a change in the spacecraft dropdown menu occurs, this function clears the S/C-specific output
            widget (which displays options pertaining to selected S/C), dynamically changes the attributes of
            current Options object and displays the options of the selected S/C.
            """
            self._out2.clear_output()

            # TODO: if user changes the spacecraft, previous values get deleted. 
            # Should maybe done so that all widgets exist all the time so that previous values 
            # are always stored, and only the output view gets changed
            # Also, the way this is done right now could cause memory leaks. (attributes do get deleted
            # but do the widget objects also?)

            with self._out2:
                if change.new == "PSP":
                    if self._sc_attrs is not None:
                        for attr in self._sc_attrs:
                            delattr(self, attr)
                    self._sc_attrs = psp_attrs
                    self.epilo_e = w.Checkbox(description="EPI-Lo electrons", value=True)
                    self.epilo_p = w.Checkbox(description="EPI-Lo protons", value=True)
                    self.epihi_e = w.Checkbox(description="EPI-Hi electrons", value=True)
                    self.epihi_p = w.Checkbox(description="EPI-Hi protons", value=True)
                    self.het_viewing = w.Dropdown(description="HET viewing", options=["A", "B"], style=style)
                    self.epilo_viewing = w.Dropdown(description="EPI-Lo viewing:", options=["F"], style=style, disabled=True, value="F")          # TODO fill in correct channels and viewings
                    self.epilo_ic_viewing = w.Dropdown(description="EPI-Lo IC viewing:", options=["T"], style=style, disabled=True, value="T")
                    self.epilo_channel = w.Dropdown(description="EPI-Lo channel", options=['3'], style=style, disabled=True, value='3')
                    self.epilo_ic_channel = w.Dropdown(description="EPI-Lo IC channel", options=['3'], style=style, disabled=True, value='3')
                    psp_vbox = w.VBox([getattr(self, attr) for attr in self._sc_attrs])
                    display(psp_vbox)

                if change.new == "SolO":
                    if self._sc_attrs is not None:
                        for attr in self._sc_attrs:
                            delattr(self, attr)
                    self._sc_attrs = solo_attrs
                    self.stix = w.Checkbox(value=True, description="STIX")
                    solo_vbox = w.VBox([getattr(self, attr) for attr in self._sc_attrs])
                    display(solo_vbox)

                if change.new == "L1 (Wind/SOHO)":
                    if self._sc_attrs is not None:
                        for attr in self._sc_attrs:
                            delattr(self, attr)
                    self._sc_attrs = l1_attrs
                    self.wind_e =  w.Checkbox(value=True, description="Wind/3DP electrons")
                    self.wind_p = w.Checkbox(value=True, description="Wind/3DP protons")
                    self.ephin = w.Checkbox(value=True, description="SOHO/EPHIN electrons")
                    self.erne = w.Checkbox(value=True, description="SOHO/ERNE protons")
                    self.ch_eph_e = w.Dropdown(description="EPHIN e channel:", options=["E150"], value="E150", disabled=True)
                    self.ch_eph_p = w.Dropdown(description="EPHIN p channel:", options=["P25"], value="P25", disabled=True)
                    self.intercal = w.BoundedIntText(value=1, min=1, max=14, description="Intercal", disabled=True)
                    l1_vbox = w.VBox([getattr(self, attr) for attr in self._sc_attrs])
                    display(l1_vbox)

                if change.new == "STEREO":
                    if self._sc_attrs is not None:
                        for attr in self._sc_attrs:
                            delattr(self, attr)
                    self._sc_attrs = stereo_attrs
                    self.sc = w.Dropdown(description="STEREO A/B:", options=["A", "B"], style=style)
                    self.sept_e = w.Checkbox(description="SEPT electrons", value=True)
                    self.sept_p = w.Checkbox(description="SEPT protons", value=True)
                    self.het_e = w.Checkbox(description="HET electrons", value=True)
                    self.het_p = w.Checkbox(description="HET protons", value=True)
                    self.sept_viewing = w.Dropdown(description="SEPT viewing", options=['sun', 'asun', 'north', 'south'], style=style)
                    
                    # TODO: "choose all energy channels" checkbox
                    self.ch_sept_e = w.SelectMultiple(description="SEPT e channels", options=range(0,14+1), rows=10, style=style)
                    self.ch_sept_p = w.SelectMultiple(description="SEPT p channels", options=range(0,29+1), rows=10, style=style)
                    self.ch_het_p =  w.SelectMultiple(description="HET p channels", options=range(0,10+1), rows=10, style=style)
                    self.ch_het_e = w.SelectMultiple(description="HET e channels", options=(0, 1, 2), value=(0, 1, 2), disabled=True, style=style)

                    stereo_vbox = w.VBox([getattr(self, attr) for attr in self._sc_attrs])
                    display(stereo_vbox)

        # TODO figure out how to have plot range update based on changes in start and end date
        # def change_plot_range(change):
            
        #     if change.owner == self.startdate:
        #         print("Changed startdate!")
        #         self.plot_range = date_selector(change.new, self.enddate.value)
        #         # dates = plot_range_interval(change.new, self.enddate.value)
                
        #     elif change.owner == self.enddate:
        #         print("Changed enddate!")
        #         self.plot_range = date_selector(self.startdate.value, change.new)
        #         # dates = plot_range_interval(self.startdate.value, change.new)
            
        #     # self.plot_range.children[0].options = dates
        #     # self.plot_range.children[0].index = (0,s len(dates) - 1)
        #     # self.plot_range.children[1]
            


        # Set observer to listen for changes in S/C dropdown menu
        self.spacecraft.observe(change_sc, names="value")

        # Likewise for start/end dates for plot range
        # self.startdate.observe(change_plot_range, names="value")
        # self.enddate.observe(change_plot_range, names='value')

        # Common attributes (dates, plot options etc.)
        self._commons = w.HBox([w.VBox([getattr(self, attr) for attr in common_attrs]), w.VBox([getattr(self, attr) for attr in variable_attrs])])
        
        # output widgets
        self._out1 = w.Output(layout=w.Layout(width="auto"))  # for common attributes
        self._out2 = w.Output(layout=w.Layout(width="auto"))  # for sc specific attributes
        self._pr_out = w.Output(layout=w.Layout(width="auto"))
        self._outs = w.HBox((w.VBox([self._out1, self._pr_out]), self._out2))         # side-by-side outputs

        display(self._outs)

        with self._out1:
            display(self._commons)

        # with self._pr_out:
        #     display(self.plot_range)
            
       

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
    
    return date_range


def load_data():
    if options.spacecraft.value == "PSP":
        psp.load_data(options)
    if options.spacecraft.value == "SolO":
        solo.load_data(options)
    if options.spacecraft.value == "L1":
        l1.load_data(options)
    if options.spacecraft.value == "STEREO":
        stereo.load_data(options)



def make_plot():
    if options.spacecraft.value == "PSP":
        return psp.make_plot(options)
    if options.spacecraft.value == "SolO":
        return solo.make_plot(options)
    if options.spacecraft.value == "L1":
        return l1.make_plot(options)
    if options.spacecraft.value == "STEREO":
        return stereo.make_plot(options)
    

options = Options()
