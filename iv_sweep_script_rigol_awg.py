#%%============================================================================
# Instrument setup
#==============================================================================
import numpy as np
import time
from tqdm import tqdm
import datetime
import pickle
import pyvisa as visa
import pandas as pd

from matplotlib import pyplot as plt
from amcc.standard_measurements.iv_sweep import run_iv_sweeps, setup_iv_sweep
from amcc.instruments.rigol_dg5000 import RigolDG5000
from amcc.instruments.RigolDG811 import RigolDG811
from amcc.instruments.lecroy_620zi import LeCroy620Zi
from amcc.instruments.switchino import Switchino

# Close all open resources
rm = visa.ResourceManager()
[i.close() for i in rm.list_opened_resources()]
#%%

lecroy_ip = '192.168.1.100'
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % lecroy_ip)
awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')
#switch = Switchino('COM7')

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
def iv_single():
    exp_iv['vpp'] = awg.get_vpp()
    exp_iv['repetition_hz'] = awg.get_freq()
    V, I = run_iv_sweeps(lecroy, num_sweeps = exp_iv['num_sweeps'], R = exp_iv['R_AWG'])
    data = dict(
        v = V,
        i = I,
        )
    return data

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
switch = Switchino('COM7')

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


#%%============================================================================
# Set IV experimental parameters
#==============================================================================
exp_iv = {}
exp_iv['test_type'] = 'IV Sweep'
exp_iv['test_name'] = 'se063'
exp_iv['R_AWG'] = 10e3
exp_iv['R_current_bias'] = 0
exp_iv['num_sweeps'] = 10
exp_iv['vpp'] = 5
exp_iv['voffset'] = 0
exp_iv['repetition_hz'] = 10

# Update vpp
awg.set_vpp(vpp = exp_iv['vpp'], channel = 1)
awg.set_voffset(exp_iv['voffset'], channel = 1)

# Update repetition rate
awg.set_freq(freq = exp_iv['repetition_hz'], channel = 1)
lecroy.set_vertical_scale(channel = 'C1', volts_per_div = exp_iv['vpp']/10)

#%%============================================================================
# Perform IV sweep one at a time w/out switch
#==============================================================================
testname = ' heater'
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%SIVsweep') + testname


data = iv_single()
df = pd.DataFrame(data)


plt.figure()
plt.plot(df['v'], df['i']*1e6)
plt.title('IV sweep' + testname)
plt.xlabel('voltage (v)')
plt.ylabel('ibias (uA)')
plt
v_lim = 0.005

df_lim = df[(df['v'] < v_lim) & (df['v'] > -v_lim)]
ibias = df_lim.loc[:,'i']
voltage = df_lim.loc[:,'v']
ib = ibias.to_numpy()
i_switch = ((max(ib)-min(ib))/2)*1e6#currently in uA
m, b  = np.polyfit(voltage, ibias, 1)
r_series = 1/m
print("Iswitch: %0.1f uA" %i_switch)
print("Rseries: %0.1f ohm" %r_series)

df.to_csv(filename + '.csv')
plt.savefig(filename + '.png', dpi = 300)
#%%============================================================================
# Perform IV sweep for each port on switch 1
#==============================================================================
ports = [1]


share_axes = True
reset_vertical_scale = False
title = None

#Run experment
data = {}
for port in tqdm(ports):
    data[port] = iv(port)
    
    if reset_vertical_scale:
        lecroy.find_vertical_scale(channel = 'C2')
        time.sleep(7)

# Convert to dataframe
import pandas as pd
df = pd.concat([pd.DataFrame({'port':port, 'i':d[0], 'v':d[1]}) for port,d in data.items()])

########### Plot data
#fig = plt.figure(); fig.set_size_inches([16,8])
if share_axes:
    fig, sub_plots = plt.subplots(1,1,sharex=share_axes, sharey=share_axes, figsize = [16,8])
#                               row,col
else:
    fig, sub_plots = plt.subplots(1,1, figsize = [16,8])
sub_plots = np.ravel(sub_plots) # Convert 2D array of sub_plots to 1D
for n, (port, iv_data) in enumerate(data.items()):
    V, I = iv_data
#    plt.subplot(2,5,port)
    sub_plots[port-1].plot(V*1e3, I*1e6,'.')
    sub_plots[port-1].set_title('Port %s' % port)
    sub_plots[port-1].set_xlabel('Voltage (mV)')
    sub_plots[port-1].set_ylabel('Current (uA)')
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
for port, d in data.items():
    v = d[0]
    i = d[1]
    iclip = i[(v<cutoff_voltage) & (v>-cutoff_voltage)]
    ic = (max(iclip)-min(iclip))/2*1e6
    print('Port %s: %0.1f uA' % (port,ic))
    

########### #%% Calculate resistance of IV curves
cutoff_current = 10e-6
cutoff_voltage = 20e-3
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
# Perform IV sweep vs current for each port on switch 1
#==============================================================================

port_pairs = [
        (1,2),
        (3,4),
        (5,6),
        (7,8),
        (9,10),
#        (2,1),
#        (4,3),
#        (6,5),
#        (8,7),
#        (10,9),
        ]
currents = np.linspace(-600e-6, 600e-6, 11)
title = 'Flex cable 5'
share_axes = True

#ports = [1,3,5,7,9]
data = []
for port_pair in tqdm(port_pairs):
    for i in currents:
        switch.select_ports(port_pair)
        V, I = iv_vs_current(i)
        d = {
                'V': V,
                'I': I,
                'port_pair' : port_pair,
                'current' : i,
                }
        data.append(d)


# Plot data
#fig = plt.figure(); fig.set_size_inches([16,8])
if share_axes:  f, sub_plots = plt.subplots(2,5,sharex=True, sharey=True, figsize = [16,8])
else:           f, sub_plots = plt.subplots(2,5, figsize = [16,8])
sub_plots = np.ravel(sub_plots) # Convert 2D array of sub_plots to 1D
for n, d in enumerate(data):
#    plt.subplot(2,5,port)
    port_pair = d['port_pair']
    sub_plots[port_pair[0]-1].plot(d['V']*1e3, d['I']*1e6,'.')
    sub_plots[port_pair[0]-1].set_title('Port %s' % [port_pair])
    sub_plots[port_pair[0]-1].set_xlabel('Voltage (mV)')
    sub_plots[port_pair[0]-1].set_ylabel('Current (uA)')
f.tight_layout()
f.suptitle(title); f.subplots_adjust(top=0.88) # Add supertitle over all subplots

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S IV sweep vs current')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.savefig(filename)


#%%============================================================================
# Perform IV vs current for several port pairs
#==============================================================================
# (,),
port_pairs = [
        (3,4),
        (5,6),
        (7,8),
        (9,10),
        ]
currents = np.linspace(0, 50e-6, 6)


data = {port_pair : {i:iv_vs_current_vs_port(i, port_pair) for i in currents} for port_pair in tqdm(port_pairs)}
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S IV sweep')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

# Plot
for pp, d in data.items():
    fig = plt.figure(); fig.set_size_inches([12,8])
    plt.axhline(0, color='lightgray')
    plt.axvline(0, color='lightgray')
    for i, iv_data in d.items():
        V, I = iv_data
        plt.plot(V*1e3, I*1e6,'.', label = 'Ibias = %0.1f uA' % (i*1e6))
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Current (uA)')
    plt.title('Ports (%s, %s)' % pp)
    
    plt.legend()

    fig_filename = filename + (' Ports %s+%s' % pp)
    pickle.dump(fig, open(fig_filename + '.fig.pickle', 'wb'))
    plt.savefig(fig_filename)
    
#%%============================================================================
# Measure resistance around zero voltage
#==============================================================================

voltage_range = 20e-3

#for n in range(len(V_list)):
#    V = V_list[n]
#    I = I_list[n]
mask = (V < voltage_range) & (V > -voltage_range)
p = np.polyfit(V[mask], I[mask], deg = 1)
resistance = 1/p[0]
print('Resistance around zero %0.2f Ohm)' % (resistance))
    
    
    

#%% OPTIONAL:  Set up SRS-based high-impedance IV curve

from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928

vs_slot = 4
dmm_slot = 7
dmm_channel = 2

dmm = SIM970('GPIB0::4', dmm_slot)
vs = SIM928('GPIB0::4', vs_slot)

dmm.set_impedance(gigaohm=True, channel = dmm_channel)
dmm.set_impedance(gigaohm=True, channel = 4)


def run_iv_sweep_srs(voltages, R_series, delay = 0.75):
    vs.reset()
    vs.set_output(True)
    time.sleep(2)
    V = []
    I = []
    for v in voltages:
        vs.set_voltage(v)
        time.sleep(delay)
        v1 = dmm.read_voltage(channel = 4)
        v1 = v
        v2 = dmm.read_voltage(channel = dmm_channel)
        V.append(v2)
        I.append((v1-v2)/R_series)
    vs.set_voltage(0)
    return np.array(V),np.array(I)

def iv(port, voltages, R_series, delay = 0.75):
    if port is not None:
        switch.select_port(port, switch = 1)
    V, I = run_iv_sweep_srs(voltages, R_series, delay = delay)
    return V, I



