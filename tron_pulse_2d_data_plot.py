#%%
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
import datetime
from matplotlib import pyplot as plt
import pyvisa as visa
import itertools
#%%To check connections to instruments

rm = visa.ResourceManager()
print(rm.list_resources())


#%%============================================================================
# Define needed functions
#==============================================================================
def parameter_combinations(parameters_dict):
    for k,v in parameters_dict.items():
        try: v[0]
        except: parameters_dict[k] = [v]
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]

#reset awgsin/ awgpulse, should do this between runs, could include at end of measurement functions
def reset_2x_awg_pulse_ktron_experiment(
    pulse_rate = 100,
):
    awgsin.reset()
    awgpulse.reset()
    counter.reset()
    time.sleep(0.1)
    sin_bias_period = 1/pulse_rate # Period of sine wave, in seconds
    num_pulses_per_period = 1
    
    #Setup counter
    counter.basic_setup()
    counter.set_impedance(ohms = 50, channel = 1)
    counter.setup_timed_count(channel = 1)
    counter.set_100khz_filter(False, channel = 1)
    counter.set_trigger(trigger_voltage = 0.05, slope_positive = True, channel = 1) #trigger set to 50 mV

    # Setup heater-pulse AWGpulse
    num_samples_delay = 511
    num_samples_write = 1
    marker_data =  [0] + [1]*num_samples_write + [0]*(num_samples_delay-1)
    voltage_data = [-1] + [1]*num_samples_write + [-1]*(num_samples_delay-1)
    marker_data = marker_data*num_pulses_per_period
    voltage_data = voltage_data*num_pulses_per_period
    awgpulse.create_waveform(voltages = voltage_data, filename = 'temp.wfm', clock = None,
                            marker1_data = marker_data, auto_fix_sample_length = False)
    awgpulse.load_file('temp.wfm')
    awgpulse.set_vhighlow(vlow = 0, vhigh = 0.1)
    awgpulse.set_marker_vhighlow(vlow = 0, vhigh = 1)
    awgpulse.set_lowpass_filter(None, channel = 1)

    #awgw.set_trigger_mode(continuous_mode=True)
    awgpulse.set_trigger_mode(trigger_mode=True)
    awgpulse.set_output(True)

    # Setup sine-bias AWG
    awgsin.set_mode(fg_mode = True)
    awgsin.set_lowpass_filter(freq = 20e6, channel = 1)
    awgsin.set_function_waveform(waveform = 'sinusoid', channel = 1)
    awgsin.set_phase(90, channel = 1)
    awgsin.set_trigger_mode(continuous_mode=True)
    awgsin.set_clock(freq = 1/sin_bias_period) # Waveform is 1000 samples long, so 1000/1e5 = 10 ms period
    awgsin.set_vhighlow(vlow = 0, vhigh = 0.1) # Inputting to resistor, so set to 1/2 value
    awgsin.set_output(True, channel = 1)

#2d-map measurement/ values
def pulse_response_2d_awg(
        tp = 2e-9, #pulse width
        vbias = 0.5,
        rbias = 10e3,
        vp = 0.1, #pulse height
        att_db = 20,
        count_time = 0.1, #counter time
        counter_trigger_voltage = 0.05,
        **kwargs,
        ):
    #Compute necessary parameters
    ibias = vbias/rbias
    vp_into_cryostat = vp*10**(-att_db/20)
    power = (vp_into_cryostat**2)/20
    energy = power*tp
    
    #Set up AWG-pulse
    awgpulse.set_clock(1/tp)
    awgpulse.set_vpp(abs(vp))
    awgpulse.set_voffset(vp/2)
    
    #Set up AWG-sine
    awgsin.set_vpp(vbias/2)
    awgsin.set_voffset(vbias/4)
    
    #Set up counter
    counter.set_trigger(counter_trigger_voltage , slope_positive = True, channel = 1)
    
    #Pause to make sure all settings are entered
    time.sleep(100e-3)
    
    #Take the data
    counts = counter.timed_count(count_time)
    
    #Reset instruments
    #reset_2x_awg_pulse_ktron_experiment(pulse_rate = 100,)
    #time.sleep(100e-3)
    #record data
    data = dict(
        vbias = vbias,
        rbias = rbias,
        ibias = ibias,
        vp = vp,
        tp = tp,
        att_db = att_db,
        vp_into_cryostat = vp_into_cryostat,
        power = power,
        energy = energy,
        counts = counts,
        counter_trigger_voltage = counter_trigger_voltage,
        count_time = count_time,
        **kwargs,
        )
    
    return data

#plots 2d pulse data
def plot_pulse_response_2d(data, max_count = 4):
    #plot 2D (Pulse voltage) vs (Pulse length), color of pixels = count#
    df = data
    for vbias, df1 in df.groupby('vbias'):
        ibias = vbias/df1['rbias'].unique()[0]
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_yscale('log')
        dfp = df.pivot('vp_into_cryostat', 'tp', 'counts')
        im = ax.pcolor(dfp.columns, np.abs(dfp.index), dfp, vmin = 0, vmax = max_count)
        fig.colorbar(im)
        plt.xlabel('Pulse width (s)')
        plt.ylabel('Pulse amplitude  (V)')
        plt.title('Pulse input response (sample %s)\nIbias = %0.1f uA' % (testname, ibias*1e6))
        plt.tight_layout()
        filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-') + testname+ (' %0.1f uA' % (ibias*1e6))
        plt.savefig(filename + '.png', dpi = 300)



def plot_1d_energy_vs_bias(df, threshold = 0.5, ylim = [0.5e-18, 1e-15]):
    df3 = df
    rbias = df3['rbias'].unique()[0]
    imin, imax = df3.vbias.min()/rbias, df3.vbias.max()/rbias
    fig, ax = plt.subplots()
    plt.xlim([imin*1e6,imax*1e6])
#    ax.set_xscale('log')
    ax.set_yscale('log')
    for t, df2 in df3.groupby('tp'):
        x = []
        y = []
        for vbias, df in df2.groupby('vbias'):
            energy_in = np.array(df.energy)
            output = np.array(df.counts)
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
    # plt.ylim(ylim)
    plt.ylabel('Minimum energy input required (J)')
    plt.title('Pulse input response')
    plt.legend()
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-')
#        pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
    plt.savefig(filename + '.png')
        
#min energy per ibias to get a click plot/ data

#make a function that finds where counts > 0
def find_v_min(vp_list, counts):    #returns min v_p value where we get a count > 0 at least twice in a row
    
    v_count = []
    for i in range(len(vp_list)-1):    #make a list of nonzero count values
        if counts[0] > 1:
            v_min = np.nan
        elif (counts[i] > 0) & (counts[i+1] >0): 
            v_count.append(vp_list[i])
    if not v_count:
        v_min = np.nan
    else:
        v_min = min(v_count)
    return v_min 
#Gets all values plus values needed for graph
def energy_min_per_ibias_values(
        tp = 2e-9,
        count_time = 0.1,
        rbias = 10e3,
        att_db = 20,
        vbias = np.linspace(0.1,0.5,10),
        vp = np.geomspace(0.1, 2,50),
        counter_trigger_voltage = 0.05,
        **kwargs):
        
    #Setup counter:
    counter.set_trigger(counter_trigger_voltage , slope_positive = True, channel = 1)
    time.sleep(0.1)
    #Main data storage
    data_main = []
    data_graph = []
    for vb in tqdm(vbias):
        
        #Compute necessary parameters
        ibias = vb/rbias
        
        #Set up AWG-sine
        awgsin.set_vpp(vb/2)
        awgsin.set_voffset(vb/4)
        time.sleep(50e-3)
        
        #list needed for min value when counts occurr
        counts_per_ib = []
        vp_per_ib = []
        
        for v in vp:
            #Set up AWG-pulse
            awgpulse.set_clock(1/tp)
            awgpulse.set_vpp(abs(v))
            awgpulse.set_voffset(v/2)
            time.sleep(50e-3)
            
            vp_into_cryostat = v*10**(-att_db/20)
            counts = counter.timed_count(count_time)
            counts_per_ib.append(counts)
            vp_per_ib.append(vp_into_cryostat)
            
            
            data = dict(
                tp = tp,
                rbias = rbias,
                vbias = vb,
                ibias = ibias,
                att_db = att_db,
                counts = counts,
                vp = v,
                vp_into_cryostat = vp_into_cryostat, 
                counter_trigger_voltage = counter_trigger_voltage,
                **kwargs)
            data_main.append(data)
            
        vp_into_cryostat_min = find_v_min(vp_per_ib, counts_per_ib)
        energy_min = (vp_into_cryostat_min**2/50)*tp
        
        data_min = dict(ibias = ibias,
                        vp_into_cryostat_min = vp_into_cryostat_min,
                        energy_min = energy_min,
                        **kwargs)
        data_graph.append(data_min)
        
    return data_main, data_graph
        
def energy_min_per_ibias_plot(data_graph):
    #list of values we will be plotting
    data_graph['ibias'] = data_graph['ibias']*1e6
    
    plt.plot(data_graph['ibias'], data_graph['energy_min'], marker = '.')
    plt.yscale('log')
    plt.xlabel('ibias (uA)')
    plt.ylabel('energy (J)')
    plt.title('%s Min required energy' %(testname))
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-') + testname+ 'min_energy'
    plt.savefig(filename + '.png', dpi = 300)
    
    
#%%============================================================================
# Setup instruments
#==============================================================================

awgpulse = TektronixAWG610('GPIB0::1')
awgsin = TektronixAWG610('GPIB0::23') # Sine generator
counter = Agilent53131a('GPIB0::10::INSTR')
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')

reset_2x_awg_pulse_ktron_experiment(pulse_rate=100)
time.sleep(0.1)
#%%============================================================================
# Normal pulse 2d map experiment/ plot
#==============================================================================
#reset instruments 1st

testname = 'code_test_A20A26'
#parameter combos lowest variable changes the fastest
parameter_dict = dict(
        vbias = [0.1], #Can only do one ibias at a time
        rbias = 10e3,
        vp = np.geomspace(0.1,2,41), #pulse height
        tp = np.geomspace(1e-9,1e-7,41), #pulse width
        att_db = 20,
        count_time = 0.1, #counter time
        counter_trigger_voltage = 0.05,
    )

#Create combinations
parameter_combos = parameter_combinations(parameter_dict)
data_list = []

for p_d in tqdm(parameter_combos):
    data_list.append(pulse_response_2d_awg(**p_d))

#Save the data 
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ktron_2d_pulse ') + testname
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')

#Plot data, saving of image in function
plot_pulse_response_2d(df, max_count=4)

#%%============================================================================
# Minimum required energy to get a click as fcn of ibias
#==============================================================================

device = 'A20A26'
#parameter combos lowest variable changes the fastest
parameter_dict = dict(
        vbias = np.linspace(0.1,1.5,41), #Can only do one ibias at a time
        rbias = 10e3,
        vp = np.geomspace(0.1,2,51), #pulse height
        tp = 2e-9, #pulse width
        att_db = 20,
        count_time = 0.1, #counter time
        counter_trigger_voltage = 0.1,
        device = [device],
    )

#Create combinations
parameter_combos = parameter_combinations(parameter_dict)
data_list = []

for p_d in tqdm(parameter_combos):
    data_list.append(pulse_response_2d_awg(**p_d))

#Save the data 
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ktron_min_energy') + testname
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')

plot_1d_energy_vs_bias(df, threshold = 0.5, ylim = [0.5e-18, 1e-15])

#%%


    
        