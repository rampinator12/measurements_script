
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

#%%============================================================================
# Setup instruments
#==============================================================================

#
awgpulse = TektronixAWG610('GPIB0::1')
awgsin = TektronixAWG610('GPIB0::23') # Sine generator
counter = Agilent53131a('GPIB0::10::INSTR')

# Setup counter
# Setup parameters
counter.reset()
time.sleep(0.1)
counter.basic_setup()
counter.set_impedance(ohms = 50, channel = 1)
counter.setup_timed_count(channel = 1)
counter.set_100khz_filter(False, channel = 1)
counter.set_trigger(trigger_voltage = 0.05, slope_positive = True, channel = 1)

t = 2e-8
# Count number of pulses for a set period of time
counter.timed_count(0.5)
# Set clock on the pulsing AWG to control the duration of the pulse
awgpulse.set_clock(1/t)
# Set peak-to-peak voltage on the pulsing AWG to set the amplitude
awgpulse.set_vpp(abs(.4))

#%% for loop to "scan' 2d plot of vp vs tp and see if we get a pulse
awgsin.set_vpp(abs(0.2))  #set ibias to 10uA
t_count = 0.2
dB = 10
Rb = 10e3
vpp_tot = []
t_tot = []


vpp_list = np.geomspace(0.1,2,50)
t_list = np.geomspace(250e-12,10e-6,50)
counts = []
awgpulse.set_vpp(abs(0))
#%%scan through voltage and time pulses, across times for fixed voltage
for v in vpp_list:
    for t in t_list:
        awgpulse.set_vpp(abs(v))
        awgpulse.set_clock(1/t)
        time.sleep(.1)
        a = counter.timed_count(t_count)
        counts.append(a)
        vpp_tot.append(v)
        t_tot.append(t)
        

#%% convert vin_pp to vout w/ 40 db attenuation =>> divide vp_tot by 100, and by 2 for amplitude not pp
vp_tot = []
for i in range(len(vpp_tot)): 
    vp_tot.append(vpp_tot[i]/(10**(dB/20)))
    
#%%2d plot with colormap for the ticks
plt.scatter(t_tot, vp_tot, c = counts, rasterized = True, vmin = 0, vmax = 4)
plt.colorbar()
plt.suptitle('A26heaterA20measured')
plt.title('Ibias = 10 uA, 10db attenuation')
plt.xlabel('Pulse width (s)')
plt.ylabel('Puse amplitude (V)')
plt.xscale('log')
plt.yscale('log')

#%%save data in csv
stack = np.stack((t_tot, vpp_tot, vp_tot,counts))
data = np.transpose(stack)
df = pd.DataFrame(data, columns = ['tp', 'vp-actual','vp','counts'])
df['count_time'] = t_count
df['Rbias'] = Rb
df['attenuation_db'] = dB
df['vp_sin'] = 0.2 
df.to_csv(r'C:\Users\dsr1\se062\pulse_tests\pulse_test_A26A20.csv')



#%% Ktron sensitivity vs bias current, take vertival scans for a fixed vp at different ib values, 
# find where 100% count rate occurs , set tp =2ns

t_p = 2e-9
Rb = 10e3
dB = 40
i_b = np.linspace(10e-6, 40e-6,30)
vpp_sin = i_b*(2*Rb) #2 for vpp (2x amplitude)
vpp_pulse = np.geomspace(0.1,2,50)

#%%
#make a function that finds where counts > 0

def find_v_min(vp_list, counts):    #returns min v_p value where we get a count > 0 at least twice in a row
    
    v_count = []
    for i in range(len(vp_list)-1):       #make a list of nonzero count values
        if (counts[i] > 0) & (counts[i+1] >0): 
            v_count.append(vp_tot[i])
    v_min = min(v_count)
    return v_min 


#%%get final list values for i_b and vp_min
vp_tot = []
counts_tot = []
i_b_tot =[]
vp_min = []   #where final pairs of i_b and v_p will be appended to 
i_b_list =[]
awgsin.set_vpp(0.2)     #set ib =10 uA
awgpulse.set_clock(1/t_p)   #set freq to 1/t_p

for i in range(len(vpp_sin)):  #scan through amplitudes and for each scan vp and find min vp value for counts with find_v_min
    
    awgsin.set_vpp(abs(vpp_sin[i]))
    time.sleep(0.3)
    counts_per_ib = []
    vp_per_ib =[]
    for v in vpp_pulse:                         #50 pt list that function checks per ib, full list of raw data in _tot
        awgpulse.set_vpp(abs(v))
        time.sleep(0.2)
        a = counter.timed_count(0.2)
        counts_tot.append(a)
        counts_per_ib.append(a)
        vp_tot.append(v/(10**(dB/20)))
        vp_per_ib.append(v/(10**(dB/20)))
        i_b_tot.append(i_b[i])
    i_b_list.append(i_b[i])
    vp_min.append(find_v_min(vp_per_ib, counts_per_ib))
    
print(i_b_list, vp_min)

#%%quick graph see if we are in right ball park

E_min = []
i_b_uA = []
for v in vp_min:
    
    E_min.append((v**2)*t_p/50)

for i in i_b_list:
    i_b_uA.append(i*1e6)
    
plt.scatter(i_b_uA, E_min)
plt.plot(i_b_uA, E_min)
plt.yscale('log')
plt.xlabel('Ibias (uA)')
plt.ylabel('minimum energy required (J)')
plt.suptitle('Pulse input response A19heaterA17measured')
plt.title('tp = 2ns')
#%%save as csv file
vp_actual = []

for v in vp_tot:
    vp_actual.append(v*200)

stack2 = np.stack((vp_actual, vp_tot, counts_tot))
data = np.transpose(stack2)

df = pd.DataFrame(data, columns = ['v_pactual', 'v_p', 'counts'])
df['count_time'] = 0.2
df['t_p'] = t_p
df['Rb'] = Rb
df['attenuation_dB'] = 40 

df.to_csv(r'C:\Users\dsr1\se062\sensitivity_vs_ibias/E_min_vs_ibias_A19A17.csv')
#%%
x = 1
y = 2
z = 'hello'

# Experiment always outputs this
data = dict(
    x = x,
    y = y,
    z = z,
        )


#data_list.append(data)
#%%
plt.scatter(t_list, counts)
plt.xscale('log')
