# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:16:51 2021

@author: vacnt
"""

#%%============================================================================
# Instrument setup
#==============================================================================
import numpy as np
import time
from tqdm import tqdm
import datetime
import pickle
import visa

from matplotlib import pyplot as plt
from amcc.standard_measurements.iv_sweep import run_iv_sweeps, setup_iv_sweep
from amcc.instruments.rigol_dg5000 import RigolDG5000
from amcc.instruments.RigolDG811 import RigolDG811
from amcc.instruments.lecroy_620zi import LeCroy620Zi
from amcc.instruments.switchino import Switchino

# Close all open resources
rm = visa.ResourceManager()
[i.close() for i in rm.list_opened_resources()]

lecroy_ip = '192.168.1.100'
lecroy = LeCroy620Zi("TCPIP::%s::inst0::INSTR" % lecroy_ip)
awg = RigolDG811('USB0::0x1AB1::0x0643::DG8A223102115::INSTR')
switch = Switchino('COM9')

#PPMS set-up
# Change Spyder directory to C:\Users\vacnt\National Institute of Standards and Technology (NIST)\Sources and Detectors - Documents\CNT\code\labview\PPMS
import os
import sys
import time
from amcc.instruments.qdinstrument import QdInstrument

dll_path = os.path.dirname(r'C:\Users\vacnt\National Institute of Standards and Technology (NIST)\Sources and Detectors - Documents\CNT\code\labview\PPMS\\')
sys.path.append(dll_path)
x = QdInstrument('DynaCool', '0.0.0.0', remote = False)



#%%============================================================================
# Initialize instruments
#==============================================================================
lecroy.reset()
time.sleep(0.5)
setup_iv_sweep(lecroy, awg, vpp = 1, repetition_hz = 10, trigger_level = 0,
               num_datapoints = 10e3, trigger_slope = 'Positive')

# awg.setup_dc(voffset = 0.0, channel = 2)


#%%============================================================================
# # Helper functions
# =============================================================================

def awg_set_current(i):
    awg.set_voffset(i*exp_iv['R_current_bias'], channel = 2)

def iv(port = None):
    if port is not None:
        switch.select_port(port, switch = 1)
    exp_iv['vpp'] = awg.get_vpp()
    exp_iv['repetition_hz'] = awg.get_freq()
    V, I = run_iv_sweeps(lecroy, num_sweeps = exp_iv['num_sweeps'], R = exp_iv['R_AWG'])
    return V, I

def iv_vs_current(i):
    awg.set_output(True, channel = 2)
    awg_set_current(i)
    time.sleep(0.1)
    V, I = iv()
    awg.set_output(False, channel = 2)
    return (V,I)


def iv_vs_current_vs_port(i, port_pair):
    switch.select_ports(port_pair)
    return iv_vs_current(i)

def plot_iv_vs_port(data):
    if share_axes:
        fig, sub_plots = plt.subplots(2,5,sharex=True, sharey=True, figsize = [16,8])
    else:
        fig, sub_plots = plt.subplots(2,5, figsize = [16,8])
    sub_plots = np.ravel(sub_plots) # Convert 2D array of sub_plots to 1D
    for n, (port, iv_data) in enumerate(data.items()):
        V, I = iv_data
    #    plt.subplot(2,5,port)
        sub_plots[port-1].plot(V*1e3, I*1e6,'.')
        sub_plots[port-1].set_title('Port %s' % port)
        sub_plots[port-1].set_xlabel('Voltage (mV)')
        sub_plots[port-1].set_ylabel('Current (uA)')
    fig.tight_layout()
    fig.suptitle(title); fig.subplots_adjust(top=0.88) # Add supertitle over all subplots
    
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S IV sweep')
    pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
    pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
    fig.savefig(filename)



#%%============================================================================
# Set vertical scale
#==============================================================================
lecroy.find_vertical_scale(channel = 'C1')
lecroy.find_vertical_scale(channel = 'C2')


#%%============================================================================
# Quick port select
#==============================================================================
#from instruments.switchino import Switchino
switch = Switchino('COM8')

switch.select_port(1, switch = 1)
switch.select_port(2, switch = 1)
switch.select_port(3, switch = 1)
switch.select_port(4, switch = 1)
switch.select_port(5, switch = 1)
switch.select_port(6, switch = 1)
switch.select_port(7, switch = 1)
switch.select_port(8, switch = 1)
switch.select_port(9, switch = 1)
switch.select_port(10, switch = 1)
switch.disable(switch = 1)

#%%============================================================================
# Set IV experimental parameters
#==============================================================================
exp_iv = {}
exp_iv['test_type'] = 'IV Sweep'
exp_iv['test_name'] = 'se063'
exp_iv['R_AWG'] = 10e3
exp_iv['R_current_bias'] = 0
exp_iv['num_sweeps'] = 10
exp_iv['vpp'] = 1
exp_iv['voffset'] = 0
exp_iv['repetition_hz'] = 20

# Update vpp
awg.set_vpp(vpp = exp_iv['vpp'], channel = 1)
awg.set_voffset(exp_iv['voffset'], channel = 1)

# Update repetition rate
awg.set_freq(freq = exp_iv['repetition_hz'], channel = 1)
lecroy.set_vertical_scale(channel = 'C1', volts_per_div = exp_iv['vpp']/10)



#%%============================================================================
# Perform IV sweep for each port on switch 1
#==============================================================================
#ports = list(range(5,2,-1))


#%%============================================================================
# Initialize instruments
#==============================================================================
lecroy.reset()
time.sleep(0.5)
setup_iv_sweep(lecroy, awg, vpp = 1, repetition_hz = 10, trigger_level = 0,
               num_datapoints = 10e3, trigger_slope = 'Positive')

# awg.setup_dc(voffset = 0.0, channel = 2)


#%%============================================================================
# # Helper functions
# =============================================================================

def awg_set_current(i):
    awg.set_voffset(i*exp_iv['R_current_bias'], channel = 2)

def iv(port = None):
    if port is not None:
        switch.select_port(port, switch = 1)
    exp_iv['vpp'] = awg.get_vpp()
    exp_iv['repetition_hz'] = awg.get_freq()
    V, I = run_iv_sweeps(lecroy, num_sweeps = exp_iv['num_sweeps'], R = exp_iv['R_AWG'])
    return V, I

def iv_vs_current(i):
    awg.set_output(True, channel = 2)
    awg_set_current(i)
    time.sleep(0.1)
    V, I = iv()
    awg.set_output(False, channel = 2)
    return (V,I)


def iv_vs_current_vs_port(i, port_pair):
    switch.select_ports(port_pair)
    return iv_vs_current(i)

def plot_iv_vs_port(data):
    if share_axes:
        fig, sub_plots = plt.subplots(2,5,sharex=True, sharey=True, figsize = [16,8])
    else:
        fig, sub_plots = plt.subplots(2,5, figsize = [16,8])
    sub_plots = np.ravel(sub_plots) # Convert 2D array of sub_plots to 1D
    for n, (port, iv_data) in enumerate(data.items()):
        V, I = iv_data
    #    plt.subplot(2,5,port)
        sub_plots[port-1].plot(V*1e3, I*1e6,'.')
        sub_plots[port-1].set_title('Port %s' % port)
        sub_plots[port-1].set_xlabel('Voltage (mV)')
        sub_plots[port-1].set_ylabel('Current (uA)')
    fig.tight_layout()
    fig.suptitle(title); fig.subplots_adjust(top=0.88) # Add supertitle over all subplots
    
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S IV sweep')
    pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
    pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
    fig.savefig(filename)



#%%============================================================================
# Set vertical scale
#==============================================================================
lecroy.find_vertical_scale(channel = 'C1')
lecroy.find_vertical_scale(channel = 'C2')


#%%============================================================================
# Quick port select
#==============================================================================
#from instruments.switchino import Switchino
switch = Switchino('COM8')

switch.select_port(1, switch = 1)
switch.select_port(2, switch = 1)
switch.select_port(3, switch = 1)
switch.select_port(4, switch = 1)
switch.select_port(5, switch = 1)
switch.select_port(6, switch = 1)
switch.select_port(7, switch = 1)
switch.select_port(8, switch = 1)
switch.select_port(9, switch = 1)
switch.select_port(10, switch = 1)
switch.disable(switch = 1)

#%%============================================================================
# Set IV experimental parameters
#==============================================================================
exp_iv = {}
exp_iv['test_type'] = 'IV Sweep'
exp_iv['test_name'] = 'se063'
exp_iv['R_AWG'] = 10e3
exp_iv['R_current_bias'] = 0
exp_iv['num_sweeps'] = 1
exp_iv['vpp'] = 2
exp_iv['voffset'] = 0
exp_iv['repetition_hz'] = 20

# Update vpp
awg.set_vpp(vpp = exp_iv['vpp'], channel = 1)
awg.set_voffset(exp_iv['voffset'], channel = 1)

# Update repetition rate
awg.set_freq(freq = exp_iv['repetition_hz'], channel = 1)
lecroy.set_vertical_scale(channel = 'C1', volts_per_div = exp_iv['vpp']/10)


#%%============================================================================
# CURVES: taking IV curves at varying temperatures
#==============================================================================

#%%IV curves CONTINUOUS sweep

ports = 9 #list(range(1,11,1))
switch.select_port(9, switch = 1)
share_axes = True
reset_vertical_scale = False
title = None
 
#Run experment
data = {}
x.setTemperature(11.5,0.5)
for n in range(10):
    temp = x.getTemperature()[1]
    data[temp] = iv(9)
    print(n)
    time.sleep(17)
if reset_vertical_scale:
    lecroy.find_vertical_scale(channel = 'C2')
    time.sleep(7)

# Convert to dataframe
import pandas as pd
df = pd.concat([pd.DataFrame({'temperature':temp, 'i':d[0], 'v':d[1]}) for temp,d in data.items()])

########### Plot data
#fig = plt.figure(); fig.set_size_inches([16,8])
if share_axes:
    fig, sub_plots = plt.subplots(2,5,sharex=share_axes, sharey=share_axes, figsize = [16,8])
#                               row,col
else:
    fig, sub_plots = plt.subplots(2,5, figsize = [16,8])
sub_plots = np.ravel(sub_plots) # Convert 2D array of sub_plots to 1D
for n, (temperature, iv_data) in enumerate(data.items()):
    V, I = iv_data

#    plt.subplot(2,5,port)
    sub_plots[n].plot(V*1e3, I*1e6 ,'.')
    sub_plots[n].set_title('temperature %0.3f K' % temperature)
    sub_plots[n].set_xlabel('Voltage (mV)')
    sub_plots[n].set_ylabel('Current (uA)')
fig.tight_layout()
if title is not None:
    fig.suptitle(title); fig.subplots_adjust(top=0.88) # Add supertitle over all subplots

########### Save data
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S IV sweep')
df.to_csv(filename + '.csv')
#pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
fig.savefig(filename)
switch.disable()

########### #%% Calculate Ic of IV curves
cutoff_voltage = 20e-3
ramp_rate = (exp_iv['vpp']/exp_iv['R_AWG']*2)/(1/exp_iv['repetition_hz']/2)
print('Ramp rate of %s A/s' % ramp_rate)
for temperature, d in data.items():
    v = d[0]
    i = d[1]
    iclip = i[(v<cutoff_voltage) & (v>-cutoff_voltage)]
    ic = (max(iclip)-min(iclip))/2*1e6
    print('temperature %0.1f K: %0.1f uA' % (temperature,ic))
#%%
#%%
ports = 9 #list(range(1,11,1))
switch.select_port(9, switch = 1)
share_axes = True
reset_vertical_scale = False
title = None
temps = [7.2,7.6,8,8.4,8.8]

#Run experment
data = {}
t = x.getTemperature()[1]
for temp in tqdm(temps):
    x.setTemperature(temp, 10)
    while t != temp:
        t = round(x.getTemperature()[1],3)
        time.sleep(1)
    print('ready')
    for n in range(30):
        print(x.getTemperature()[1])
        time.sleep(1)
   
    data[temp] = iv(9)
    
if reset_vertical_scale:
    lecroy.find_vertical_scale(channel = 'C2')
    time.sleep(7)

# Convert to dataframe
import pandas as pd
df = pd.concat([pd.DataFrame({'temperature':temp, 'i':d[0], 'v':d[1]}) for temp,d in data.items()])

########### Plot data
#fig = plt.figure(); fig.set_size_inches([16,8])
if share_axes:
    fig, sub_plots = plt.subplots(2,2,sharex=share_axes, sharey=share_axes, figsize = [16,8])
#                               row,col
else:
    fig, sub_plots = plt.subplots(2,2, figsize = [16,8])
sub_plots = np.ravel(sub_plots) # Convert 2D array of sub_plots to 1D
for n, (temperature, iv_data) in enumerate(data.items()):
    V, I = iv_data
#    plt.subplot(2,5,port)
    sub_plots[n].plot(V*1e3, I*1e6 ,'.')
    sub_plots[n].set_title('temperature %0.1f K' % temperature)
    sub_plots[n].set_xlabel('Voltage (mV)')
    sub_plots[n].set_ylabel('Current (uA)')
fig.tight_layout()
if title is not None:
    fig.suptitle(title); fig.subplots_adjust(top=0.88) # Add supertitle over all subplots

########### Save data
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S IV sweep')
df.to_csv(filename + '.csv')
#pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
fig.savefig(filename)
switch.disable()

########### #%% Calculate Ic of IV curves
cutoff_voltage = 20e-3
ramp_rate = (exp_iv['vpp']/exp_iv['R_AWG']*2)/(1/exp_iv['repetition_hz']/2)
print('Ramp rate of %s A/s' % ramp_rate)
for temperature, d in data.items():
    v = d[0]
    i = d[1]
    iclip = i[(v<cutoff_voltage) & (v>-cutoff_voltage)]
    ic = (max(iclip)-min(iclip))/2*1e6
    print('temperature %0.1f K: %0.1f uA' % (temperature,ic))
    
#%%
########### #%% Calculate resistance of IV curves
cutoff_current = 1000e-6
cutoff_voltage = 2000e-3
for port, d in data.items():
    v = d[0]
    i = d[1]
    # Select only rising edge of waveform so retrapping parts of IV are not measured
    p = 0 # padding
    selection1  = slice(len(v)//2+p,len(v)*3//4-p)
    selection2  = slice(p,len(v)//4-p)
    v = np.concatenate([v[selection1],v[selection2]])
    i = np.concatenate([i[selection1],i[selection2]])
    selection = (i<cutoff_current) & (i>-cutoff_current) & (v<cutoff_voltage) & (v>-cutoff_voltage)
    i = i[selection]
    v = v[selection]
    try:
        idx = np.isfinite(v) & np.isfinite(i)
        p = np.polyfit(i[idx],v[idx],1)
    except: p= [-1]
    print('Port %s: %0.1f Ohm' % (port,p[0]))
    
#%%============================================================================
# IC VALUES SWEEPS, NOT RECORDING WHOLE CURVE
#==============================================================================
#%%============================================================================
# Initialize instruments
#==============================================================================

lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')
#awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')
awg = RigolDG811('USB0::0x1AB1::0x0643::DG8A223102115::INSTR')

switch = Switchino('COM8')

lecroy.reset()
awg.reset()
time.sleep(5)

setup_ic_measurement_lecroy(lecroy, vpp = 10, repetition_hz = 200,
                     trigger_level = 10e-3, trigger_slope = 'Positive',
                     coupling_ch1 = 'DC1M', coupling_ch2 = 'DC1M')
lecroy.set_horizontal_scale(20e-3/10.0, time_offset = 0)
lecroy.set_trigger_mode('Auto')
lecroy.set_memory_samples(num_samples = 10e3)

awg.set_load('INF')
awg.setup_ramp(freq=100, vpp=1, voffset=0, symmetry_percent = 50)
awg.set_output(True)
awg.set_vhighlow(vhigh = 4, vlow = 0, channel = 1)

#awg.setup_arb_wf(t = [0,1,1.5,9.5,10], v = [0,0,1,1,0], channel = 2)
#awg.set_freq(freq = 200, channel = 2)
#awg.set_vhighlow(vhigh = 0.01, vlow = 0, channel = 2)
#awg.set_output(True, channel = 2)

#awg.align_phase()


#%%============================================================================
# # Helper functions
# =============================================================================
def awg_set_output(output = False):
    awg.set_output(output, channel = 2)
    awg.align_phase()

def awg_set_current(i):
    awg.set_vhighlow(vlow = 0, vhigh = i*exp_ic['R_current_bias'], channel = 2)

def measure_ic(port = None):
    if port is not None:
        switch.select_port(port, switch = 1)
    vpp = awg.get_vpp()
    repetition_hz = awg.get_freq()
    R_AWG = exp_ic['R_AWG']
    voltage_data = run_ic_sweeps(lecroy, num_sweeps = exp_ic['num_sweeps'])
    ic_data = voltage_data/R_AWG
    ic_median = np.median(ic_data)
    ic_std = np.std(ic_data)
    return locals()
    

def ic_vs_current(i):
    awg_set_current(i)
    
    time.sleep(0.1)
    ic_data = measure_ic()
    ic_data.update({'i':i})
    ic_median = ic_data['ic_median']
    ic_std = ic_data['ic_std']
    print('Current value %0.2f uA  -  Median Ic = %0.2f uA / Std. dev Ic = %0.2f uA' % (i*1e6, ic_median*1e6, ic_std*1e6))
    return ic_data

def ic_vs_current_vs_port(i, port_pair):
    switch.select_ports(port_pair)
    data = ic_vs_current(i)
    data['port_pair'] = port_pair
    return data
    
def ic_string(ic_data):
    ic_median = ic_data['ic_median']
    ic_std = ic_data['ic_std']
    return 'Median Ic = %0.2f uA / Std. dev Ic = %0.2f uA' % (ic_median*1e6, ic_std*1e6) + \
    ' (Ramp rate %0.3f A/s (Vpp = %0.1f V, rate = %0.1f Hz, R = %0.1f kOhms))' \
        % (calc_ramp_rate(ic_data['vpp'], ic_data['R_AWG'], ic_data['repetition_hz'], 'RAMP'), ic_data['vpp'], ic_data['repetition_hz'], ic_data['R_AWG']/1e3)


#%%============================================================================
# Quick port select
#==============================================================================
#from instruments.switchino import Switchino
#switch = Switchino('COM7')

switch.select_port(1, switch = 1)
switch.select_port(2, switch = 1)
switch.select_port(3, switch = 1)
switch.select_port(4, switch = 1)
switch.select_port(5, switch = 1)
switch.select_port(6, switch = 1)
switch.select_port(7, switch = 1)
switch.select_port(8, switch = 1)
switch.select_port(9, switch = 1)
switch.select_port(10, switch = 1)
switch.disable(switch = 1)

switch.select_port(1, switch = 2)
switch.select_port(2, switch = 2)
switch.select_port(3, switch = 2)
switch.select_port(4, switch = 2)
switch.select_port(5, switch = 2)
switch.select_port(6, switch = 2)
switch.select_port(7, switch = 2)
switch.select_port(8, switch = 2)
switch.select_port(9, switch = 2)
switch.select_port(10, switch = 2)
switch.disable(switch = 2)

switch.disable()

switch.select_ports((3,4))

#%%============================================================================
# Setup experimental variables
#==============================================================================
exp_ic = {}
exp_ic['test_type'] = 'Ic Sweep'
exp_ic['test_name'] = 'None'
exp_ic['R_AWG'] = 10e3
exp_ic['R_current_bias'] = 1e3
exp_ic['num_sweeps'] = 1000

# Update vpp
exp_ic['vpp'] = 1.5
awg.set_vhighlow(vhigh = exp_ic['vpp'], vlow = 0, channel = 1)

# Update repetition rate
exp_ic['repetition_hz'] = 1000
awg.set_freq(freq = exp_ic['repetition_hz'], channel = 1)
#awg.set_freq(freq = exp_ic['repetition_hz'], channel = 2)
#awg.align_phase()


#%%============================================================================
# Quick Ic Test
#==============================================================================
data = measure_ic()
print(ic_string(data))
#### quick save
#filename = 'SE005 Device C4' + ' Ic Sweep'
#time_str = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
#scipy.io.savemat(filename  + '.mat', mdict={'ic_data':ic_data})

#%% Plot histogram
plt.hist(data['ic_data']*1e6, bins = 50, color = 'g')
plt.xlabel('Ic (uA)')
plt.show()


#%%Temperature sweep

data_list = []
temperature = -1
x.setTemperature(13,1)
while temperature < 12:
    data = measure_ic()
    temperature = x.getTemperature()[1]
    data['temperature'] = temperature
    print(temperature)
    data_list.append(data)


df = pd.DataFrame(data_list)
import datetime
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S temperature sweeps ')
df.to_csv(filename+'.csv')

ic = df['ic_median']
temperature = df['temperature']
plt.plot(temperature,np.array(ic)*1e6, '.:')
plt.title('Ic vs T (amplified)')
plt.xlabel('Temperature (K)')
plt.ylabel('Ic (uA)')
plt.savefig(filename + '.png', dpi = 300)
#%%Frequency measurement
frequency = np.geomspace(10,30e3,50)
data_list = []
for f in frequency:
    # Update repetition rate
    print(f) 
    # Update repetition rate
    exp_ic['repetition_hz'] = f
    awg.set_freq(freq = exp_ic['repetition_hz'], channel = 1)
    #awg.set_freq(freq = exp_ic['repetition_hz'], channel = 2)
    #awg.align_phase()

    data = measure_ic()
    
    temperature = x.getTemperature()[1]
    data['temperature'] = temperature   
    data['frequency'] = f
    data_list.append(data)
df = pd.DataFrame(data_list)    
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S frequency Ic test')
df.to_csv(filename+'.csv')

ic = df['ic_median']
frequency = df['frequency']

plt.semilogx(frequency,np.array(ic)*1e6, '.:')
plt.title('Ic vs frequency (amplified), T = 1.7k')
plt.xlabel('frequency (Hz)')
plt.ylabel('Ic (uA)')
plt.savefig(filename + '.png', dpi = 300)

#%%
x.setTemperature(6.5,10)
time.sleep(10)
temp = x.getTemperature()[1]

data_list = []

x.setTemperature(7,10)
while temp != 7.000:
    temp = round(x.getTemperature()[1],3)
    time.sleep(1)
    print('not ready')
print('ready!')
t0 = time.time()
for i in range(50):
    data = measure_ic() 
    t = time.time() - t0
    temperature = x.getTemperature()[1]
    data['temperature'] = temperature   
    data['time'] = t
    data_list.append(data)
    print(i)   
df = pd.DataFrame(data_list) 
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S con')
df.to_csv(filename+'.csv')

ic = df['ic_median']
icerr = df['ic_std']

plt.title('continous Ic at 7K')
plt.plot(df['time'], np.array(ic)*1e6, ':.')
#plt.errorbar(df['time'], np.array(ic)*1e6, yerr =icerr*1e6 )
plt.xlabel('time (s)')
plt.ylabel('Ic (uA)')
plt.savefig(filename + '.png', dpi = 300)

#%%Long sweep where each temp value is checked before taking a measurement
temps = np.linspace(1.7,12,400)
data_list = []
for t in temps:
    x.setTemperature(t,10)
    temp = x.getTemperature()[1]
    while temp != round(t,3):
        temp = round(x.getTemperature()[1],3)
        time.sleep(1)
    print(t)
    data = measure_ic()
    temperature = x.getTemperature()[1]
    data['temperature'] = temperature
    print(temperature)
    data_list.append(data)

df = pd.DataFrame(data_list) 
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S con')
df.to_csv(filename+'.csv')
ic = df['ic_median']
temperature = df['temperature']
plt.plot(temperature,np.array(ic)*1e6, '.:')
plt.title('Ic vs T')
plt.xlabel('Temperature (K)')
plt.ylabel('Ic (uA)')
plt.savefig(filename + '.png', dpi = 300)
