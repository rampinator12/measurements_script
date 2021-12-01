# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:43:57 2021

@author: vacnt
"""

#%%

# Ic measurement code
# Run add_path.py first
import pyvisa as visa
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

lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')
awg = RigolDG811('USB0::0x1AB1::0x0643::DG8A223102115::INSTR')
dmm = SIM970('COM9', 3)
vs = SIM928('COM9', 2)
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
exp_ic['R_current_bias'] = 10e3
exp_ic['num_sweeps'] = 100

# Update vpp
exp_ic['vpp'] = 2
awg.set_vhighlow(vhigh = exp_ic['vpp'], vlow = 0, channel = 1)

# Update repetition rate
exp_ic['repetition_hz'] = 100
awg.set_freq(freq = exp_ic['repetition_hz'], channel = 1)
#awg.set_freq(freq = exp_ic['repetition_hz'], channel = 2)
#awg.align_phase()


#%%============================================================================
# Quick Ic Test
#==============================================================================
data = measure_ic()
print(ic_string(data))

#%%============================================================================
#Run experiment
#==============================================================================
device = 3.5
voltages = np.geomspace(0.01,0.6,51)
heater_width = 0.2e-6
rbias = 1e3
data_list = []

vs.set_voltage(0)
initial_ic = measure_ic()
ic_0 = initial_ic['ic_median']

for v in voltages:
    
    vs.set_voltage(v)
    v2 = dmm.read_voltage(channel = 2)
    
    power = v2**2/50
    heater_area_um = (heater_width*1e6)**2
    power_density_nW_um2 = (power*1e9)/(heater_area_um)
    time.sleep(0.1)
    measurement = measure_ic()
    
    
    ic_median = measurement['ic_median']
    ic_std = measurement['ic_std']
    
    
    data = dict(
        device = device,
        voltage_set_DC = v,
        v_heater = v2,
        rbias = rbias,
        power = power,
        heater_width = heater_width,
        power_density_nW_um2 = power_density_nW_um2,
        ic_median = ic_median,
        ic_std = ic_std,
        
        )
    data_list.append(data)

vs.set_voltage(0)
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S IV sweep')
plt.figure()

df= pd.DataFrame(data_list)
#df = pd.read_csv(r)
plt.semilogx(df['power_density_nW_um2'], np.array(df['ic_median'])*1e6, ':.')
plt.xlabel('power density (nW/um^2)')
plt.ylabel('ic (uA)')
plt.title('Device %s' %device)

df.to_csv(filename + 'ic_heater_power.csv')
plt.savefig(filename+ 'ic_heater_power.csv.png', dpi = 300)











