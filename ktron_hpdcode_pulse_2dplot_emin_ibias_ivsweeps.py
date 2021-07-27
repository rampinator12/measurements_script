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
def pulse_response_2d_plot(data, max_count = 4):
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
    reset_2x_awg_pulse_ktron_experiment(pulse_rate=100)
    time.sleep(0.1)
    #Setup counter:
    counter.set_trigger(counter_trigger_voltage , slope_positive = True, channel = 1)
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

# I-V curves for nanowire/ heater
def v_in_stack(volt_lim, num_pts):
    a = np.linspace(0,volt_lim,num_pts)
    b = np.linspace(volt_lim,-volt_lim,2*num_pts)
    c = np.linspace(-volt_lim,0,num_pts)
    v_in = np.concatenate((a,b,c))
    return v_in

def iv_sweep(
        t_delay = 0.5,
        rbias = 10e3,
        v_in = 1,
        channel1 = 1,
        channel2 = 2,
        ):
    # set voltage, wait t_delay, then take measurement
    vs.set_voltage(v_in)
    time.sleep(t_delay)
    
    v1 = dmm.read_voltage(channel = channel1) #reading before bias resistor
    v2 = dmm.read_voltage(channel = channel2) # reading after bias resistor
    ibias = (v1-v2)/rbias
    
    data = dict(
        rbias = rbias,
        v_in = v_in,
        ibias = ibias,
        v_plot = v2
        )
    return data

def steadystate_iv_sweep( # note heater is set bias then we sweep other side. We are taking Iv sweep of 'other side' as heater is fixed at same biased current
        t_delay = 0.5,
        rbias = 10e3,
        rheater = 10e3,
        v_in_heater = 1,
        v_in_nanowire = 1,
        channel1_heater = 3,
        channel2_heater = 4,
        channel1_nanowire = 1,
        channel2_nanowire = 2,
        ):
    
    #set heater bias first
    vsh.set_voltage(v_in_heater)
    time.sleep(0.1)
    v1h = dmmh.read_voltage(channel = channel1_heater) #above rheater
    v2h = dmmh.read_voltage(channel = channel2_heater) #below rheater
    iheater = (v1h - v2h)/rheater
    
    #now set the nanowire bias (what we would "sweep" in param combinations)
    vs.set_voltage(v_in_nanowire)
    time.sleep(0.1)
    v1 = dmmh.read_voltage(channel = channel1_nanowire)
    v2 = dmmh.read_voltage(channel = channel2_nanowire)
    ibias = (v1-v2)/rbias
    
    data = dict(
        rbias = rbias,
        rheater = rheater,
        v_in_heater = v_in_heater,
        v_in_nanowire = v_in_nanowire,
        iheater = iheater,
        ibias = ibias,
        v_nanowire_plot = v2, #always plot this side
        )
        
    return data

#%%============================================================================
# Setup instruments IV sweeps with/ without heater bias
#==============================================================================
#Chanage numbers to correct ports!

dmm = SIM970('GPIB0::4', 7) #dmm are voltage outputs (read), vs are applied voltage, h for heater other is nanowire
vs = SIM928('GPIB0::4', 6)

dmmh = SIM970('GPIB0::4', 8)
vsh = SIM928('GPIB0::4', 5)

#%%============================================================================
# I-V Sweep 1d (no heater biased, testing single pads)
#==============================================================================
#zero out voltage 
vs.set_voltage(0)
vs.set_output(True)
time.sleep(0.5)

#Make combos (only v in this case) still nice to see progress bar
testname = 'enter port/ device name here'
parameter_dict = dict(
    t_delay = 0.5,   
    rbias = 10e3,
    v_in = v_in_stack(volt_lim = 5, num_pts = 50),
    channel1 = 1, #Change channels accoardingly 1 means above resistor
    channel2 = 2,
    )
#create combos
parameter_combos = parameter_combinations(parameter_dict)
data_list = []

for p_d in tqdm(parameter_combos):
    data_list.append(iv_sweep(**pd))
    
#save the data
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ktron_2d_pulse ') + testname
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')

#plot the data
plt.plot(df['v_in'], df['ibias']*1e6, marker = '.')
plt.title('IV sweep %s' %testname)
plt.xlabel('Voltage (v)')
plt.ylabel('ibias (uA)')
plt.savefig(filename + '.png')


#%%============================================================================
# I-V Steady State Sweep (fixed biased on one side, IV sweep of other side, in code 'heater' has fixed bias and 'nanowire' has full sweep)
#==============================================================================
#zero out voltage
vs.set_voltage(0)
vsh.set_voltage(0)
vs.set_output(True)
vsh.set_output(True)
time.sleep(0.1)

#Make combos
testname = 'enter test name here'
parameter_dict = dict(
    v_in_heater = np.linspace(0,5,10), #having this high will sweep through this first
    v_in_nanowire = v_in_stack(volt_lim = 5, num_pts = 50),
    t_delay = 0.5,
    rbias = 10e3,
    rheater = 10e3,
    channel1_heater = 3, #change these accoardingly 1 meas above resistor
    channel2_heater = 4,
    channel1_nanowire = 1,
    channel2_nanowire = 2,
    )

#create combos
parameter_combos = parameter_combinations(parameter_dict)
data_list = []

for p_d in tqdm(parameter_combos):
    data_list.append(iv_sweep(**pd))
    
#save the data
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ktron_2d_pulse ') + testname
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')

#plot the data.. gotta figure out the grouby method

for name, gd in df.groupby(['iheater']):
    plt.plot(gd.v_nanowire_plot, gd.ibias *1e6, marker = '.', label = 'iheater = %0.1f uA' %name[0]*1e6)
plt.legend()
plt.xlabel('voltage (V)')
plt.ylabel('ibias (uA')
plt.title('Steady State IV ' + testname)
plt.savefig(filename+ '.png')


    
#%%============================================================================
# Setup instruments for pulse tests
#==============================================================================

awgpulse = TektronixAWG610('GPIB0::1')
awgsin = TektronixAWG610('GPIB0::23') # Sine generator
counter = Agilent53131a('GPIB0::10::INSTR')
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')

#%%============================================================================
# Normal pulse 2d map experiment/ plot
#==============================================================================
#reset instruments 1st
reset_2x_awg_pulse_ktron_experiment(pulse_rate=100)
time.sleep(1e-3)

testname = 'A17A18'
#parameter combos lowest variable changes the fastest
parameter_dict = dict(
        tp = np.geomspace(4e-10,1e-7,50), #pulse width
        vbias = [1.1], #Can only do one ibias at a time
        rbias = 10e3,
        vp = np.geomspace(0.1,2,50), #pulse height
        att_db = 50,
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

#Plot data, saving of image in function2
pulse_response_2d_plot(df, max_count=4)

#%%============================================================================
# Minimum required energy to get a click as fcn of ibias
#==============================================================================
reset_2x_awg_pulse_ktron_experiment(pulse_rate = 100)
testname = 'change this to sample name'
#Take the data, change the arguments for required parameters
data_main, data_graph = energy_min_per_ibias_values(
        tp = 2e-9,
        count_time = 0.1,
        rbias = 10e3,
        att_db = 20,
        vbias = np.linspace(0.1,0.5,10),
        vp = np.geomspace(0.1, 2,50),
        counter_trigger_level = 0.05,)

dfmain = pd.DataFrame(data_main)
dfgraph = pd.DataFrame(data_graph)

#save the data
filename_main = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-') + testname+ 'min_energy_main'
dfmain.to_csv(filename_main + '.csv') 
filename_graph = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-') + testname+ 'min_energy_graph'
dfgraph.to_csv(filename_graph + '.csv')

#Make/ save the plot
energy_min_per_ibias_plot(dfgraph)