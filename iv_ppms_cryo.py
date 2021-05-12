#%%
import pyvisa as visa
import amcc
from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


rm = visa.ResourceManager()
print(rm.list_resources())

#Set 6 for bias, 5 for heater. dmm/vs for bias dmmh/vsh for heater
dmm = SIM970('GPIB0::4', 7)
vs = SIM928('GPIB0::4', 6)

dmmh = SIM970('GPIB0::4', 8)
vsh = SIM928('GPIB0::4', 5)



#%%

# =============================================================================
# Parameters
# =============================================================================
num_pts = 50
delay = 0.75
volt_lim = 5


# =============================================================================
# Experiment
# =============================================================================
vs.set_voltage(0)
vs.set_output(True)
time.sleep(delay)
a = np.linspace(0,volt_lim,num_pts)
b = np.linspace(volt_lim,-volt_lim,2*num_pts)
c = np.linspace(-volt_lim,0,num_pts)
v_in = np.concatenate((a,b,c))
v_device =[]
I_device= []

I = I_device
V = v_device
port = 20
Rb = 100e3
Rb_array = Rb*np.ones(len(I))
port_array = port*np.ones(len(I))

#%%
# =============================================================================
# Graph data
# =============================================================================

port_array =[]
Rb_array = []
v1_array = []
v2_array = []
v_input = []
i_b = []


for j in range(len(v_in)):
        v1 = dmm.read_voltage(channel =1)
        v2 = dmm.read_voltage(channel = 2)
        I = ((v1-v2)/Rb)*1e6
        a = v_in[j]
        vs.set_voltage(a)  
        time.sleep(.75)
        v_input.append(v_in[j])
        v1_array.append(v1)
        v2_array.append(v2)
        port_array.append(port)
        Rb_array.append(Rb)
        i_b.append(I)

plt.plot(v2_array, i_b)
plt.title("A20")
plt.xlabel("V (volts)")
plt.ylabel("I_b(uA)")
# =============================================================================
# Save data
# =============================================================================
data = np.stack((port_array, Rb_array, v2_array, i_b))
data_format = np.transpose(data)
df = pd.DataFrame(data_format, columns = ['port','Rb','V_d','I_b'])
df.to_csv(r'C:\Users\dsr1\se062\DSR_iv_sweep_cryo_8.csv')

#%%
# =============================================================================
# i-v funtion with set bias (A17 = bias A19 = heater)
# =============================================================================
#parameters
num_pts = 15
delay = 0.75
volt_lim = 5

#start with vb = 2.5 V
Rb = 100e3
Rh = 100e3

vs.set_voltage(0)
vsh.set_voltage(0)
vs.set_output(True)
vsh.set_output(True)
time.sleep(delay)

a = np.linspace(0,volt_lim,num_pts)
b = np.linspace(volt_lim,-volt_lim,2*num_pts)
c = np.linspace(-volt_lim,0,num_pts)
v_in = np.concatenate((a,b,c))

#set vb to 3.5 v and calculate ib
vs.set_voltage(2.5)
v1b = dmm.read_voltage(channel = 1)
v2b = dmm.read_voltage(channel = 2)
ib = (v1b-v2b)/Rb



port_array =[]
Rh_array = []
v1h_array = []
v2h_array = []
vh_input = []
i_h = []
i_b = []
v_b = np.linspace(0,3.5,14)

def take_iv(v_bias):
    vs.set_voltage(0)
    vsh.set_voltage(0)
    time.sleep(delay)
    
    vs.set_voltage(v_bias)
    time.sleep(delay)
    v1 = dmm.read_voltage(channel = 1)
    v2 = dmm.read_voltage(channel = 2)
    ibias = (v1-v2)/Rb
    
    for j in range(len(v_in)):
            v1h = dmm.read_voltage(channel = 3)
            v2h = dmm.read_voltage(channel = 4)
            I = ((v1h-v2h)/Rh)
            a = v_in[j]
            vsh.set_voltage(a)  
            time.sleep(delay)
            vh_input.append(v_in[j])
            v1h_array.append(v1h)
            v2h_array.append(v2h)
            port_array.append(port)
            Rh_array.append(Rh)
            i_h.append(I)
            i_b.append(ibias)
    return 

for j in range(len(v_b)):
    take_iv(v_b[j])
    

plt.plot(v2h_array, i_h)
plt.title("trial")
plt.xlabel("V (volts)")
plt.ylabel("I_b(A)")
#%%

#PPMS i_v

from amcc.instruments.switchino import Switchino
switch = Switchino('COM8')

port_array =[]
Rb_array = []
v1_array = []
v2_array = []
v_input = []
i_b = []

for i in range(1,11):
    switch.select_port(port = i)
    port = i
    Rb = 20e3
   
    
    for j in range(len(v_in)):
        v1 = dmm.read_voltage(channel =1)
        v2 = dmm.read_voltage(channel = 2)
        I = ((v1-v2)/Rb)*1e6
        a = v_in[j]
        vs.set_voltage(a)  
        time.sleep(.75)
        v_input.append(v_in[j])
        v1_array.append(v1)
        v2_array.append(v2)
        port_array.append(port)
        Rb_array.append(Rb)
        i_b.append(I)
         

data = np.stack((port_array,Rb_array,v_input,v1_array,v2_array,i_b))
data_format = np.transpose(data)
df = pd.DataFrame(data_format, columns = ['port','R(ohms)', 'V_in(volts)','V1(volts)','V2(volts)', 'I_b(uA)' ])
df.to_csv(r'C:\Users\vacnt\Documents\GitHub\amcc-measurement\amcc\DSR_iv_sweep_fullch3.csv')
switch.disable()


#%%
#PPMS switchino curve
fig, axs = plt.subplots(2,5, sharex = True, sharey = True)
axs_flat = axs.flatten()

df = pd.read_csv(r'C:\Users\vacnt\Documents\GitHub\amcc-measurement\amcc\DSR_iv_sweep_fullch3.csv')

for i in range(1,11):
    port = i
    df_subset = df[df['port'] == port]
    axs_flat[port-1].plot(df_subset['V2(volts)'], df_subset['I_b(uA)'])
for ax in axs.flat:
    ax.set(xlabel='voltage (V)', ylabel='I_bias(uA)')
fig.set_size_inches(16,8)
fig.savefig('DSR_iv_sweep_ch3.png')
#%

