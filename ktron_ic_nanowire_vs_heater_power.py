# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:43:57 2021

@author: vacnt
"""

#%%

# Ic measurement code
# Run add_path.py first
import pyvisa as visa
from amcc.instruments.rigol_dg5000 import RigolDG5000
from amcc.instruments.RigolDG811 import RigolDG811
from amcc.instruments.lecroy_620zi import LeCroy620Zi
from amcc.instruments.switchino import Switchino
from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928
from amcc.standard_measurements.ic_sweep import setup_ic_measurement_lecroy, run_ic_sweeps, calc_ramp_rate
#from standard_measurements.general_function_vs_port import measure_vs_parameter, measure_vs_parameter_vs_ports
#from useful_functions.save_data_vs_param import *
from tqdm import tqdm # Requires "colorama" package also on Windows to display properly
import numpy as np
import time
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import datetime

# Close all open resources
rm = visa.ResourceManager()
[i.close() for i in rm.list_opened_resources()]
    

#%%============================================================================
# Initialize instruments
#==============================================================================

lecroy_ip = '192.168.1.100'
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % lecroy_ip)
awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')
dmm = SIM970('GPIB0::4', 7)
vs = SIM928('GPIB0::4', 4)
#switch = Switchino('COM7')

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
def measure_ic_single():
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
exp_ic['R_current_bias'] = 10e3
exp_ic['num_sweeps'] = 1000

# Update vpp
exp_ic['vpp'] = 2
awg.set_vhighlow(vhigh = exp_ic['vpp'], vlow = 0, channel = 1)

# Update repetition rate
exp_ic['repetition_hz'] = 1000
awg.set_freq(freq = exp_ic['repetition_hz'], channel = 1)
#awg.set_freq(freq = exp_ic['repetition_hz'], channel = 2)
#awg.align_phase()

#%%============================================================================
# Quick Ic Test
#==============================================================================
vs.set_voltage(0)
time.sleep(0.5)
data = measure_ic_single()
print(ic_string(data))
x, y = lecroy.get_wf_data(channel = 'F2')

plt.figure()
plt.plot(np.array(x)*1e2,np.array(y),'.')
plt.xlabel('ic nanowire (uA)')
plt.ylabel('frequency')
plt.savefig('2022-12.png', dpi = 300)

#%%============================================================================
#Run experiment: Ic of nanowire as a fucntion of DC heater power
#==============================================================================
device = 5.7
volt = np.geomspace(0.01,2,101)
volt_reverse = np.flip(volt)
voltages = np.concatenate((volt,volt_reverse))

heater_width = 0.2e-6
rbias = 1e3
data_list = []

vs.set_voltage(0)
initial_ic = measure_ic_single()
ic_0 = initial_ic['ic_median']

for v in voltages:
                    #Set up heater bias/ calculate power into heater
    vs.set_voltage(v)
    v1 = dmm.read_voltage(channel = 1)
    v2 = dmm.read_voltage(channel = 2)
    ibias = (v1-v2)/rbias
    r_heater = v2/ibias
    power = v2*ibias
    heater_area_um = (heater_width*1e6)**2
    power_density_nW_um2 = (power*1e9)/(heater_area_um)
    time.sleep(0.1)
    measurement = measure_ic_single()
    
    #measure ic of nanowire
    ic_median = measurement['ic_median']
    ic_std = measurement['ic_std']
    
    
    data = dict(
        device = device,
        voltage_set_DC = v,
        v_heater = v2,
        rbias = rbias,
        r_heater = r_heater,
        ibias = ibias,
        power = power,
        heater_width = heater_width,
        power_density_nW_um2 = power_density_nW_um2,
        ic_median = ic_median,
        ic_std = ic_std,
        
        )
    data_list.append(data)
#Save data/ Plot curves
vs.set_voltage(0)
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S IV sweep')

df= pd.DataFrame(data_list)



plt.figure()
power_density = np.array(df['power_density_nW_um2'])
ic_meas = np.array(df['ic_median'])
p_for,p_rev = np.split(power_density,2)
ic_mes_for, ic_mes_rev = np.split(ic_meas,2)

plt.semilogx(p_for, ic_mes_for*1e6, ':.', c = 'b', label = 'forward')
plt.semilogx(p_rev, ic_mes_rev*1e6, ':.', c = 'r', label = 'reverse')
plt.xlabel('power density (nW/um^2)')
plt.ylabel('ic (uA)')
plt.title('Device %s' %device)
plt.legend()

df.to_csv(filename + 'ic_heater_power.csv')
plt.savefig(filename+ 'ic_heater_power.csv.png', dpi = 300)

#%% Plotting v_heater vs ibias
plt.figure()
v_h = np.array(df['v_heater'])
ib = np.array(df['ibias'])


v_for , v_rev = np.split(v_h, 2)
ib_for, ib_rev = np.split(ib, 2)


plt.plot(ib_for*1e6, v_for, '.:', color = 'b', label = 'forward')
plt.plot(ib_rev*1e6, v_rev, '.:', color = 'r', label = 'reversed')
plt.xlabel('ibias_heater (uA)')
plt.ylabel('v_heater (V)')
plt.legend()
plt.savefig('2022-5.png', dpi = 300)

#%%Plotting heater resistance and Ic of nanowire vs power density in the same plot
fig , ax = plt.subplots()


power = np.array(df['power_density_nW_um2'])
rh = np.array(df['r_heater'])
ic_meas = np.array(df['ic_median'])

p_for , p_rev = np.split(power, 2)
rh_for, rh_rev = np.split(rh, 2)
ic_mes_for, ic_mes_rev = np.split(ic_meas,2)

ax.set_xlabel('power density (nW/um^2)')
ax.set_ylabel('ic (uA)')
ax.semilogx(p_for, ic_mes_for*1e6, ':.', c = 'b', label = 'ic_forward')
ax.semilogx(p_rev, ic_mes_rev*1e6, ':.', c = 'r', label = 'ic_reverse')
ax.tick_params(axis = 'y')

ax1 = ax.twinx()
ax1.set_ylabel('R_heater (ohm)')
ax1.semilogx(p_for, rh_for, ':.', c = 'g', label = 'rh_forward')
ax1.semilogx(p_rev, rh_rev, ':.', c = 'm', label = 'rh_reverse')
ax1.tick_params(axis = 'y')
ax1.set_ylim([3e3,7e3])

ax.legend(loc = 'lower left')
ax1.legend(loc = 'center right')
plt.savefig('2022-7.png', dpi = 400)


#%%============================================================================
#Run experiment: 
#==============================================================================

device = '3.1'

tp = 5e-9
awgpulse.set_clock(1/tp)
vp = np.linspace(0.1,2,11)

vbias = 0.40 # 35 uA
rbias = 10e3
att_db = 10
v_att = (vp/2)*10**(-att_db/20)
power = (v_att)**2/50
energy = power*tp


#set bias current
vs.set_voltage(vbias)
v1 = dmm.read_voltage(channel = 1)
v2 = dmm.read_voltage(channel = 2)
time.sleep(0.5)
ibias = (v1-v2/rbias)

#run the experiment

data_list = []
for i in range(len(vp)):
    vs.set_voltage(vbias)
    awgpulse.set_vpp(vp[i])
    awgpulse.set_voffset(vp[i]/2)
    time.sleep(0.5)
    
    v_mes = dmm.read_voltage(channel  = 2) #voltage on nanowire
    v_into_cryo = v_att[i]
    volt = vp[i]
    p = power[i]
    en = energy[i]
    
    data = dict(
        device = device,
        tp = tp,
        vbias = vbias,
        rrbias = rbias,
        ibias = ibias,
        vp = volt,
        power = p,
        energy = en,
        v_heater = v_mes,
        )
    vs.set_voltage(0)
    time.sleep(0.5)
    data_list.append(data)
    
df = pd.DataFrame(data_list)
plt.plot(df['energy'], df['v_heater'])
    
#%%

device = '3.1'

tp = 5e-9
awgpulse.set_clock(1/tp)
vp = np.linspace(0.1,2,21)
length = len(vp)
vbias = 0.35 # 35 uA
rbias = 10e3
att_db = 10



#set bias current
vs.set_voltage(vbias)
v1 = dmm.read_voltage(channel = 1)
v2 = dmm.read_voltage(channel = 2)
time.sleep(0.5)
ibias = vbias/rbias

for v in vp:
    
    awgpulse.set_vpp(v)
    awgpulse.set_voffset(v/2)
    time.sleep(0.5)
    awgpulse.set_vpp(0)
    time.sleep(2)
    
    v_heater = dmm.read_voltage(channel = 2)
    
    if v_heater > 0.1:
        v_att = (v/2)*10**(-att_db/20)
        power = (v_att)**2/50
        energy = power*tp
        
        data = dict(
            device = device,
            vbias = vbias,
            rbias = rbias,
            ibias = ibias,
            vp = v,
            tp = tp,
            att_db = att_db,
            v_into_cryo = v_att,
            power = power,
            energy = energy,
            
            )
        print(data)
        break
    else:
        pass
awgpulse.set_vpp(0)
print(data)    
    
    
   
    
#%%
    

