#%%

# Heater measurement code
# Run add_path.py first
from amcc.instruments.rigol_dg5000 import RigolDG5000
from amcc.instruments.lecroy_620zi import LeCroy620Zi
from amcc.instruments.switchino import Switchino
from amcc.instruments.tektronix_awg610 import TektronixAWG610
from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.agilent_53131a import Agilent53131a

from amcc.standard_measurements.ic_sweep import setup_ic_measurement_lecroy, run_ic_sweeps, calc_ramp_rate
from tqdm import tqdm # Requires "colorama" package also on Windows to display properly
import numpy as np
import pandas as pd
import numpy as np
import time
import pickle
import datetime
from matplotlib import pyplot as plt


import itertools
def parameter_combinations(parameters_dict):
    for k,v in parameters_dict.items():
        try: v[0]
        except: parameters_dict[k] = [v]
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]

def plot_2d_energy_map(data, max_count_ratio = 3):
    # Plot 2D (Pulse voltage)  vs (Pulse length), where color = vdmm (latching or not)
    dfall = pd.DataFrame(data)
    for ports, df2 in dfall.groupby('ports'):
        for vbias, df in df2.groupby('vbias'):
            ibias = vbias/df['rbias'].unique()[0]
            fig, ax = plt.subplots()
            ax.set_xscale('log')
            ax.set_yscale('log')
            dfp = df.pivot('vpulse_actual', 't', 'pulse_count_ratio')
            #X,Y = np.meshgrid()
            im = ax.pcolor(dfp.columns, np.abs(dfp.index), dfp, vmin = 0, vmax = max_count_ratio)
            fig.colorbar(im)
#            plt.clim(color_range)
            plt.xlabel('Pulse width (s)')
            plt.ylabel('Pulse amplitude  (V)')
            plt.title('Pulse input response (Ports %s)\nIbias = %0.1f uA' % (ports, ibias*1e6))
            plt.tight_layout()
            filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-') + ('ports ' + str(ports)) + (' %0.1f uA' % (ibias*1e6))
            plt.savefig(filename + '.png')
            pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
        #    pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
    
    


    
def plot_1d_energy_vs_bias(data, threshold = 0.5, ylim = [0.5e-18, 1e-15]):
    df4 = pd.DataFrame(data)
    rbias = df4['rbias'].unique()[0]
    imin, imax = df4.vbias.min()/rbias, df4.vbias.max()/rbias
    for ports, df3 in df4.groupby('ports'):
        fig, ax = plt.subplots()
        plt.xlim([imin*1e6,imax*1e6])
    #    ax.set_xscale('log')
        ax.set_yscale('log')
        for t, df2 in df3.groupby('t'):
            x = []
            y = []
            for vbias, df in df2.groupby('vbias'):
                energy_in = np.array(df.energy_in)
                output = np.array(df.pulse_count_ratio)
                ibias = vbias/rbias
                threshold_idx = np.argmax(output > threshold)
                # Check if it ever actually clicked, or if it always latched
                if sum(output > threshold) == 0: required_energy = np.nan
                elif sum(output > threshold) == len(output): required_energy = np.nan
                else: required_energy = energy_in[threshold_idx]
                y.append(required_energy)
                x.append(ibias)
            plt.plot(np.array(x)*1e6,y,'.:', label = ('t = %0.1f ns' % (t*1e9)))
        plt.xlabel('Ibias (uA)')
        plt.ylim(ylim)
        plt.ylabel('Minimum energy input required (J)')
        plt.title('Pulse input response - Ports %s' % str(ports))
        plt.legend()
        filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-') + ('ports ' + str(ports))
#        pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
        plt.savefig(filename + '.png')
        
        
def plot_2d_energy_vs_bias(data, max_count_ratio = 2.1):
    # Plot 2D (Pulse voltage)  vs (Pulse length), where color = vdmm (latching or not)
    dfall = pd.DataFrame(data)
    for ports, df2 in dfall.groupby('ports'):
        for t_ns, df in df2.groupby('t_ns'):
#            ibias = vbias/df['rbias'].unique()[0]
            fig, ax = plt.subplots()
#            ax.set_xscale('log')
            ax.set_yscale('log')
            dfp = df.pivot('energy_in', 'ibias', 'pulse_count_ratio')
            #X,Y = np.meshgrid()
            im = ax.pcolor(dfp.columns*1e6, dfp.index, dfp, vmin = 0, vmax = max_count_ratio)
            fig.colorbar(im)
#            plt.clim(color_range)
            plt.xlabel('Bias current (uA)')
            plt.ylabel('Pulse energy (J)')
            plt.title('Pulse input response (Ports %s)\nt = %0.1f ns' % (ports, t_ns))
            plt.tight_layout()
            filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-') + ('ports ' + str(ports)) + (' t = %0.1f ns' % (t_ns))
            plt.savefig(filename + '.png')
#            pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
        
def pulse_response_2x_awg(
    t = 50e-9,
    vbias = 0.5,
    vpulse = 0.1,
    vpulse_att_db = 40,
    rbias = 100e3,
    count_time = 0.1,
    pulse_rate = 0,
#    pulses_per_sine = 1,
    ports = None,
    **kwargs,
    ):
    
    # Switch select
    global last_ports
    if ports is not None:
        if last_ports != ports:
            switch.select_ports(port_pair = ports)
            time.sleep(1)
            last_ports = ports
    
    ibias = vbias/rbias
    t_ns = t*1e9
    
    awgw.set_clock(1/t)
#    awgw.set_vhighlow(vlow = 0, vhigh = vpulse)
    awgw.set_vpp(abs(vpulse))
    awgw.set_voffset(vpulse/2)
#    awgsin.set_vhighlow(vlow = 0, vhigh = vbias/2)
    awgsin.set_vpp(abs(vbias/2))
    awgsin.set_voffset(vbias/4)
    time.sleep(0.05)
    vpulse_actual = vpulse*10**(-vpulse_att_db/20)
    count_rate = counter.timed_count(count_time)
    pulse_count_ratio = count_rate/pulse_rate/count_time
    energy_in = vpulse_actual**2/50*t
    output = pulse_count_ratio
    
    data = locals()
    
    return data


#%%============================================================================
# Setup instruments
#==============================================================================

#
awgw = TektronixAWG610('GPIB0::1')
awgsin = TektronixAWG610('GPIB0::23') # Sine generator
#lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')
#awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')
switch = Switchino('COM7')
counter = Agilent53131a('GPIB0::10::INSTR')


#%%============================================================================
# Heater-pulse measurement setup
# 
# For latching or nonlatching devices - The bias is a sine wave, resetting 
# periodically.  When the sine wave is at its peak, it triggers the heater 
# input pulse(s) so counts can be collected on a counter.
#==============================================================================


# Connector AWG520 Marker1 output to AWG610 ext. trigger input
# Manually load 'sin.wfm' on CH1 of AWG520
# Shoudl be set, but make sure trigger on "External" on AWG610


trigger_voltage = 0.05
sin_bias_period = 10e-3 # Period of sine wave, in seconds
num_pulses_per_period = 1

# Setup counter
# Setup parameters
counter.reset()
time.sleep(0.1)
counter.basic_setup()
counter.set_impedance(ohms = 50, channel = 1)
counter.setup_timed_count(channel = 1)
counter.set_100khz_filter(False, channel = 1)
counter.set_trigger(trigger_voltage = trigger_voltage, slope_positive = True, channel = 1)

# Setup heater-pulse AWG
num_samples_delay = 511
num_samples_write = 1
marker_data =  [0] + [1]*num_samples_write + [0]*(num_samples_delay-1)
voltage_data = [-1] + [1]*num_samples_write + [-1]*(num_samples_delay-1)
marker_data = marker_data*num_pulses_per_period
voltage_data = voltage_data*num_pulses_per_period
awgw.create_waveform(voltages = voltage_data, filename = 'temp.wfm', clock = None, marker1_data = marker_data, auto_fix_sample_length = False)
awgw.load_file('temp.wfm')
awgw.set_vhighlow(vlow = 0, vhigh = 1)
awgw.set_marker_vhighlow(vlow = 0, vhigh = 1)
#awgw.set_trigger_mode(continuous_mode=True)
awgw.set_trigger_mode(trigger_mode=True)
awgw.set_output(True)


# Setup sine-bias AWG
awgsin.set_mode(fg_mode = True)
awgsin.set_lowpass_filter(20e6, channel = 1)
awgsin.set_trigger_mode(continuous_mode=True)
awgsin.set_clock(freq = 1/sin_bias_period) # Waveform is 1000 samples long, so 1000/1e5 = 10 ms period
awgsin.set_vhighlow(vlow = 0, vhigh = 0.08/2) # Inputting to resistor, so set to 1/2 value
awgsin.set_output(True, channel = 1)


pulse_rate = 1/sin_bias_period*num_pulses_per_period # Number of input pulses per second


#%% 2D mapping (latching devices)
global last_ports
last_ports = None

parameters_dict = dict(
#        ports = [(5,3), (5,2)], #   
        ports = [(2,1), ], #  (4,3), (6,5), (8,7), (10,9),
        vbias = [0],
        vpulse = np.geomspace(0.4, 1,11),
        t = 2e-9,
        vpulse_att_db = 10,
        rbias = 10e3,
        count_time = 0.1,
        pulse_rate = pulse_rate,
        ) # Lowest variable is fastest-changing index

parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))

data = []
for pc in tqdm(parameter_combos):
    print(list(pc.values()))
    data.append(pulse_response_2x_awg(**pc))
switch.disable()

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
plot_2d_energy_map(data, )



#%% 1D mapping - energy required vs bias current (latching devices)
global last_ports
last_ports = None

parameters_dict = dict(
        ports = [('A20','A26')], #,  (7,8), (9,10)
#        ports = [(2,1)], # (4,3), (6,5), (8,7), (10,9),
        vbias = np.arange(0.1, 0.8, 0.05),#np.arange(0.04,0.8,0.01),
        vpulse = np.geomspace(0.1, 0.8,41),
        t = [2e-9], # 1e-9,2e-9,4e-9
        vpulse_att_db = 10,
        rbias = 10e3,
        count_time = 0.1,
        pulse_rate = pulse_rate,
        ) # Lowest variable is fastest-changing index

parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))



data = []
for pc in tqdm(parameter_combos):
    print(list(pc.values()))
    data.append(pulse_response_2x_awg(**pc))
switch.disable()

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
plot_1d_energy_vs_bias(data, ylim = [0.5e-14, 1e-10])
plot_2d_energy_vs_bias(data)
        

#%%
threshold = 0.01

df = pd.DataFrame(data)
df['latched'] = df['vdmm']>threshold
for name, group in df.groupby('ibias'):
    plt.figure()
    plt.semilogx(group['energy_in']/1.6e-19, np.clip(group['vdmm'].values,0,2), '.', label = 'Ibias = %0.1f uA' % (name*1e6))
    plt.xlabel('Pulse energy (eV)')
    plt.ylabel('Ratio of pulse to expected pulses')
    hist, bin_edges = np.histogram(group.energy_in[group.latched == True]/1.6e-19, bins = 1000, range = [0,10000])
    onset_energy = bin_edges[np.nonzero(hist > 3)[0][0]]
    print(str(np.round(name*1e6)) + ' uA / ' + str(onset_energy) + ' eV')
    plt.ylim([0,2])
    plt.tight_layout()
    plt.legend()

#%% 

df = pd.DataFrame(data)
energy_95 = []
i = []
for name, group in df.groupby('ibias'):
    i.append(name)
    ratios = group['pulse_count_ratio'].values
    e95_arg = (np.array(ratios) > 0.95).argmax()
    e95 = group['energy_in'].values[e95_arg]
    energy_95.append(e95)
plot(np.array(i)*1e6, np.array(energy_95)/1.6e-19,'.')
xlabel('Bias current (uA)')
ylabel('Pulse energy required (eV)')


#%% Jitter measurement

lecroy.reset()
lecroy.set_display_gridmode(gridmode = 'Single')
lecroy.set_vertical_scale(channel = 'C1', volts_per_div = 0.2)
lecroy.set_vertical_scale(channel = 'C2', volts_per_div = 0.2)
lecroy.set_horizontal_scale(20e-9)
#lecroy.set_memory_samples(num_datapoints)

lecroy.set_trigger(source = 'C1', volt_level = 0.5, slope = 'Positive')
lecroy.set_trigger_mode(trigger_mode = 'Normal')
lecroy.set_coupling(channel = 'C1', coupling = 'DC50') # CH1 is input voltage readout
lecroy.set_coupling(channel = 'C2', coupling = 'DC50') # CH2 is channel voltage readout

lecroy.set_parameter(parameter = 'P1', param_engine = 'Dtime@Level',
                     source1 = 'C1', source2 = 'C2', show_table=True)
lecroy.setup_math_histogram(math_channel = 'F2', source = 'P1', num_values = 300)
lecroy.setup_math_trend(math_channel = 'F1', source = 'P1', num_values = 10e3)
lecroy.set_parameter(parameter = 'P5', param_engine = 'HistogramSdev', source1 = 'F2', source2 = None)
lecroy.set_parameter(parameter = 'P6', param_engine = 'HistogramMedian', source1 = 'F2', source2 = None)

def jitter_pulse_response_2x_awg(
    t = 2e-9,
    vbias = 0.5,
    vpulse = 0.1,
    vpulse_att_db = 40,
    rbias = 1e3,
    pulse_rate = 0,
    count_time = 3,
#    pulses_per_sine = 1,
    ports = None,
    **kwargs,
    ):
    
    # Switch select
    global last_ports
    if ports is not None:
        if last_ports != ports:
            switch.select_ports(port_pair = ports)
            time.sleep(1)
            last_ports = ports
    
    ibias = vbias/rbias
    t_ns = t*1e9
    
    awgw.set_clock(1/t)
#    awgw.set_vhighlow(vlow = 0, vhigh = vpulse)
    awgw.set_vpp(abs(vpulse))
    awgw.set_voffset(vpulse/2)
#    awgsin.set_vhighlow(vlow = 0, vhigh = vbias/2)
    awgsin.set_vpp(abs(vbias/2))
    awgsin.set_voffset(vbias/4)
    time.sleep(0.05)
    vpulse_actual = vpulse*10**(-vpulse_att_db/20)
    energy_in = vpulse_actual**2/50*t
    
    lecroy.clear_sweeps()
    time.sleep(count_time)
    x, jitter_delays = lecroy.get_wf_data(channel='F1')
    if len(jitter_delays) != 0:
        jitter_std = np.std(jitter_delays)
        jitter_median = np.median(jitter_delays)
    else:
        jitter_std = np.nan
        jitter_median = np.nan
        
    data = locals()
    return data
        
    
#    num_sweeps = int(num_sweeps)
#    no_delays = False
#    while (measured_sweeps < num_sweeps+1):
#        measured_sweeps = len(lecroy.get_wf_data(channel='F1')[0])
#        time.sleep(0.1)
#        elapsed_time += 0.1
#        if measured_sweeps == 0 and elapsed_time > 1:
#            jitter_delays = np.array([np.nan]*num_sweeps)
#            no_delays = True
#            break
#    if not no_delays:
#        x, jitter_delays = lecroy.get_wf_data(channel='F1')
#        while len(jitter_delays) < num_sweeps:
#            x, jitter_delays = lecroy.get_wf_data(channel='F1')
#            time.sleep(0.05)
#        del x
#    jitter_delays = jitter_delays[:num_sweeps]
#    jitter_std = np.std(jitter_delays)
#    jitter_median = np.median(jitter_delays)
#    del measured_sweeps, elapsed_time,no_delays
    
    
    
#%% 
global last_ports
last_ports = None

parameters_dict = dict(
#        ports = [(1,2), (3,4), (5,6), (9,10)], #,  (7,8), (9,10)
        ports = [(4,3)], # (4,3), (6,5), (8,7), (10,9),
        vbias = [0.2,0.25,0.3],#np.arange(0.04,0.8,0.01),
        vpulse = np.geomspace(0.04, 0.4,41),
        t = [2e-9], # 1e-9,2e-9,4e-9
        vpulse_att_db = 40,
        rbias = 1e3,
        count_time = 3,
        pulse_rate = pulse_rate,
        ) # Lowest variable is fastest-changing index

parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))



data = []
for pc in tqdm(parameter_combos):
    print(list(pc.values()))
    data.append(jitter_pulse_response_2x_awg(**pc))
switch.disable()

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
#plot_1d_energy_vs_bias(data)
#plot_2d_energy_vs_bias(data)

df = pd.DataFrame(data)
ylim = [0,1000]
#ax.set_yscale('log')
for name,group in df.groupby('ports'):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    for name2,group2 in group.groupby('ibias'):
        semilogx(group2.energy_in, group2.jitter_std*1e12, '.', label = str(name2*1e6) + ' uA')
    plt.xlabel('Pulse energy (J)')
    plt.ylim(ylim)
    plt.ylabel('Jitter std (ps)')
    plt.title('Jitter response to pulse input - Ports %s' % str(name))
    legend()
    plt.tight_layout()
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-') + ('ports ' + str(ports)) + (' t = %0.1f ns' % (t_ns))
    plt.savefig(filename + '.png')
