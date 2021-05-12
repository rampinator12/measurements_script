
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

vpp_tot = []
t_tot = []

vpp_list = np.geomspace(0.1,2,50)
t_list = np.geomspace(10e-10,10e-8,50)
counts = []
awgpulse.set_vpp(abs(0))
#%%scan through voltage and time pulses, across times for fixed voltage
for v in vpp_list:
    for t in t_list:
        awgpulse.set_vpp(abs(v))
        awgpulse.set_clock(1/t)
        time.sleep(.1)
        a = counter.timed_count(0.2)
        counts.append(a)
        vpp_tot.append(v)
        t_tot.append(t)

#%% convert vin_pp to vout w/ 40 db attenuation =>> divide vp_tot by 100, and by 2 for amplitude not pp
vp_tot = []
for i in range(len(vpp_tot)):
    vp_tot.append(vpp_tot[i]/200)
    
#%%2d plot with colormap for the ticks
plt.scatter(t_tot, vp_tot, c = counts, rasterized = True, vmin = 0, vmax = 4)
plt.colorbar()
plt.suptitle('A17heaterA19measured')
plt.title('Ibias = 10 uA, 40db attenuation')
plt.xlabel('Pulse width (s)')
plt.ylabel('Puse amplitude (V)')
plt.xscale('log')
plt.yscale('log')

#%%save data in csv
stack = np.stack((t_tot,vp_tot,counts))
data = np.transpose(stack)
df = pd.DataFrame(data, columns = ['tp','vp','counts'])
df['Rbias'] = 10e3
df['attenuator'] = 40
df.to_csv(r'C:\Users\dsr1\se062\pulse_tests\pulse_testA17A19.csv')
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
