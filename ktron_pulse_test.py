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
import pyvisa as visa

#%%

rm = visa.ResourceManager()
print(rm.list_resources())

#%%============================================================================
# Setup instruments
#==============================================================================

#
awgpulse = TektronixAWG610('GPIB0::1')
awgsin = TektronixAWG610('GPIB0::23') # Sine generator
counter = Agilent53131a('GPIB0::10::INSTR')
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')

#%%

# Setup counter
# Setup parameters
counter.reset()
time.sleep(0.1)
counter.basic_setup()
counter.set_impedance(ohms = 50, channel = 1)
counter.setup_timed_count(channel = 1)
counter.set_100khz_filter(False, channel = 1)
counter.set_trigger(trigger_voltage = 0.05, slope_positive = True, channel = 1) #trigger set to 50 mV

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


##Setting up sin wave (from ADAMs' code)
sin_bias_period = 10e-3 # Period of sine wave, in seconds
num_pulses_per_period = 1

# Setup sine-bias AWG
awgsin.set_mode(fg_mode = True)
awgsin.set_lowpass_filter(20e6, channel = 1)
awgsin.set_trigger_mode(continuous_mode=True)
awgsin.set_clock(freq = 1/sin_bias_period) # Waveform is 1000 samples long, so 1000/1e5 = 10 ms period
awgsin.set_vhighlow(vlow = 0, vhigh = 0.08/2) # Inputting to resistor, so set to 1/2 value
awgsin.set_output(True, channel = 1)


pulse_rate = 1/sin_bias_period*num_pulses_per_period # Number of input pulses per second


#%%============================================================================
# pulse heater measurements
#==============================================================================
#FILENAME

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + 'A20A26_50uA'
save_path = 'C:/Users/dsr1/se062/pulse_tests_ibias_correct/'
ports = 'trial123'
#parameters
vpp_sine = 0.1
awgsin.set_vpp(abs(vpp_sine)/2)  
awgsin.set_voffset(abs(vpp_sine)/4) #set vbias /2 because of R and offset by /4 so that vmin of wave always at v = 0
t_count = 0.1
dB = 20
Rb = 10e3
ib =round((vpp_sine/Rb)*1e6)

# Input values that must be set
vpp_list = np.geomspace(0.1,2,50)
t_list = np.geomspace(250e-12,10e-6,50)
counts = []
awgpulse.set_vpp(abs(0))


def pulse_2d_values():    
    vpp_tot = []
    vp_tot = []
    t_tot = []
    for v in vpp_list:
        for t in t_list:
            awgpulse.set_vpp(abs(v))
            awgpulse.set_voffset(abs(v)/2)
            awgpulse.set_clock(1/t)
            time.sleep(.1)
            a = counter.timed_count(t_count)
            counts.append(a)
            vpp_tot.append(v)
            t_tot.append(t)
    for i in range(len(vpp_tot)): 
        vp_tot.append(vpp_tot[i]/(10**(dB/20))) #convert vin_pp to vout w/ propper attenuated value
    
    stack = np.stack((t_tot, vpp_tot, vp_tot,counts))
    data_array = np.transpose(stack)
    
    return data_array
        

def pulse_2d_plot(data_array):
    
    
    plt.scatter(data_array[:,0], data_array[:,2], c = data_array[:,3], rasterized = True, vmin = 0, vmax = 4)
    plt.colorbar()
    plt.suptitle(ports)
    plt.title('Ibias =  %d uA, %d db attenuation' %(ib, dB))
    plt.xlabel('Pulse width (s)')
    plt.ylabel('Pulse amplitude (V)')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(save_path + filename + '.png')
    
    
    data = pd.DataFrame(data_array, columns = ['tp', 'vp-actual','vp','counts'])
    data['count_time'] = t_count
    data['Rbias'] = Rb
    data['attenuation_db'] = dB
    data['vp_sin'] = vpp_sine
    data.to_csv(save_path + filename + '.csv')  

#%%Running the functions and taking the measurements
a = pulse_2d_values()
pulse_2d_plot(a)



#%%============================================================================
#IHEATER PULSE TEST: fix tp and Ib (through device) change bias current through heater, call it Iheater and vary vp, make 
#similar 2d plot with vp vs Iheater
#==============================================================================

#File-Path:
    
filename1 = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + 'A11A25_ib_40uA'
save_path1 = 'C:/Users/dsr1/se062/ibias_iheater/'
sample = 'A11A25'

#Set bias current to nanowire
vpp_sine = 0.2
awgsin.set_vpp(abs(vpp_sine)/2, channel = 1)  
awgsin.set_voffset(abs(vpp_sine)/4, channel = 1)
awgsin.set_vhighlow(vlow = 0.0, vhigh = 0.3, channel = 2)

#Parameters

Rb = 10e3
ib = round((vpp_sine/Rb)*1e6)
dB = 40
t_p = 2e-9
t_count = 0.1
i_heater = np.linspace(4e-6,55e-6,20)  #set to 20x20 = 400 points, not as many as other pulse test, just trying to get
vpp_sin_heater = i_heater*Rb
vpp_pulse = np.geomspace(0.1,2,20)

awgsin.set_vhighlow(vlow = 0.0, vhigh = 0.3, channel = 2)

def iheater_pulse_2d_values():
    i_htot = []
    vsin_heater_tot = []
    vpulse_actual = []
    vpulse_tot = []
    counts = []
    
    for i in range(len(vpp_sin_heater)):
        awgsin.set_vpp(vpp_sin_heater[i]/2, channel =  2)
        awgsin.set_voffset(vpp_sin_heater[i]/4, channel = 2)
        
        for v2  in vpp_pulse:
            
            awgpulse.set_vpp(abs(v2))
            awgpulse.set_voffset(abs(v2)/2)
            awgpulse.set_clock(1/t_p)
            time.sleep(0.1)
            counts.append(counter.timed_count(t_count))
            i_htot.append(i_heater[i])
            vsin_heater_tot.append(vpp_sin_heater[i])
            vpulse_actual.append(v2)
            vpulse_tot.append(v2/(10**(dB/20)))
            
    data0 = np.stack((vpulse_tot, vsin_heater_tot, i_htot, counts, vpulse_actual))    
    data1 = np.transpose(data0)
    
    return data1
 

def iheater_pulse_2d_plot(data_array):
    
    i_heater = data_array[:,2]*1e6
     
    plt.scatter(i_heater, data_array[:,0], c = data_array[:,3], rasterized = True, vmin = 0, vmax = 4) 
    plt.colorbar()
    plt.xlabel('Iheater (uA)')
    plt.ylabel('Pulse Amplitude (V)')  
    plt.yscale('log')   
    plt.title('Ibias = %0.1f uA = %0.1f dB-Iheater vs vp sweep' %(ib,dB))
    plt.savefig(save_path1 + filename1 + '.png')

    df = pd.DataFrame(data_array, columns = ['vp', 'vsin_heater_tot', 'ih','counts', 'vp_actual'])
    df['count_time'] = t_count
    df["rb"] = Rb
    df['attenuation_db'] = dB

    df.to_csv(save_path1 + filename1 + '.csv')
    
#%%Run the two funstions 
b = iheater_pulse_2d_values()
iheater_pulse_2d_plot(b)


#%%============================================================================
#Ktron sensitivity vs bias current, take vertival scans for a fixed vp at different ib values, 
# find where 100% count rate occurs 
#==============================================================================

#FILEPATH
filename2 = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + 'emin_vs_ibias_A19A17'
save_path2 = 'C:/Users/dsr1/se062/sensitivity_vs_ibias/'
#Parameters

ports = 'A19A17'
t_p = 2e-9
t_count = 0.1
Rb = 10e3
dB = 20
i_b = np.linspace(10e-6, 50e-6,30)
vpp_sin = i_b*Rb #Set up for offset = 0.02  from adam's code no half needed!!!
vpp_pulse = np.geomspace(0.1,2,50)


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


#get final list values for i_b and vp_min

def pulse_ibias_emin():
    
    vp_actual = []
    vp_tot = []
    counts_tot = []
    i_b_tot =[]
    vp_min = []   #where final pairs of i_b and v_p will be appended to 
    i_b_list =[]
    awgpulse.set_clock(1/t_p)   #set freq to 1/t_p
    E_min = []
    
    for i in range(len(vpp_sin)):  #scan through amplitudes and for each scan vp and find min vp value for counts with find_v_min
    
        awgsin.set_vpp(abs(vpp_sin[i])/2)
        awgsin.set_voffset(abs(vpp_sin[i])/4)
        time.sleep(0.1)
        counts_per_ib = []
        vp_per_ib =[]
        for v in vpp_pulse:                         #50 pt list that function checks per ib, full list of raw data in _tot
            awgpulse.set_vpp(abs(v))
            awgpulse.set_voffset(abs(v)/2)
            time.sleep(0.1)
            a = counter.timed_count(t_count)
            counts_tot.append(a)
            counts_per_ib.append(a)
            vp_actual.append(v)
            vp_tot.append(v/(10**(dB/20)))
            vp_per_ib.append(v/(10**(dB/20)))
            i_b_tot.append(i_b[i])
        i_b_list.append(i_b[i])                             # These two lists are what we end up graphing, after vp is 
        vp_min.append(find_v_min(vp_per_ib, counts_per_ib)) # converted to emin
    
    
    for v in vp_min:
    
        E_min.append((v**2)*t_p/50)

    
    stack = np.stack((vp_actual, vp_tot, counts_tot))
    data_array = np.transpose(stack)
    stack2 = np.stack((i_b_list, vp_min, E_min))
    data_plot_array = np.transpose(stack2)
    
    return data_array, data_plot_array

#Now a function that plots and saves the data

def pulse_ibias_emin_plot(data_array1, data_array2):
    
    i_b_uA = data_array2[:,0]*1e6  #converts to uA and ns
    E_min = data_array2[:,2]
    t_ns = t_p*1e9

    
    plt.scatter(i_b_uA, E_min)
    plt.plot(i_b_uA, E_min)
    plt.yscale('log')
    plt.xlabel('Ibias (uA)')
    plt.ylabel('minimum energy required (J)')
    plt.suptitle('Pulse input response Ports' + ports)
    plt.title('tp = %0.1f ns' %(t_ns))
    plt.savefig(save_path2 + filename2 + '.png')
    
    data = pd.DataFrame(data_array1, columns = ['v_pactual', 'v_p', 'counts'])
    data['count_time'] = t_count
    data['t_p'] = t_p
    data['Rb'] = Rb
    data['attenuation_dB'] = dB
    
    data_graph = pd.DataFrame(data_array2, columns = ['ib', 'vp_min', 'Emin'])
    
    data.to_csv(save_path2 + filename2 + '.csv')
    data_graph.to_csv(save_path2 + filename2 + 'graph.csv')    #saves graph data separately from raw

#%%
c, d = pulse_ibias_emin()
pulse_ibias_emin_plot(c,d)
    
#%%
#Find heater + series resistance from IV curve

data123 = pd.read_csv(r'C:\Users\dsr1\se062\Iv_measurements\DSR_iv_sweep_cryo_8.csv')

data_graph = data123[data123['V_d']>0.02]
print(np.polyfit(data_graph['I_b'], data_graph['V_d'], 1))
#%%Finding miimum tp and vp values in a dataset

vp_line = []
tp_line = []


t_values = np.geomspace(250e-12,10e-6,50)
df = pd.read_csv(r'C:\Users\dsr1\se062\pulse_tests\pulse_test_A20A26.csv')

df_need = df[(df['counts']>0) & (df['tp']>1e-9)]
plt.scatter(df_need['tp'], df_need['vp'], c = df_need['counts'])
plt.xscale('log')
plt.yscale('log')

a = df_need['vp'].min()
b = df_need[df_need['vp'] == a]

print('tp_min=  ', b['tp'].min())
print('v_pmin = ', a)
    
#%% try to talk to lecroy
vp =.4
vp_sine = .4
awgsin.set_vpp(vp_sine/2)
awgsin.set_voffset(vp_sine/4)
awgpulse.set_vpp(vp)
awgpulse.set_voffset(vp/2)


#%%
v_in = 0.5
v_ref = 0.22
Pin = v_in**2/50
Prefl = (2*v_ref)**2/50

print('ratio = %s' % ())




#%%

# Setup lecroy
from amcc.instruments.lecroy_620zi import LeCroy620Zi
lecroy_ip = '192.168.1.100'
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % lecroy_ip)

#%%

t_background,v_background = lecroy.get_wf_data(channel = 'F1')

#%%
t,v = lecroy.get_wf_data(channel = 'F1')
figure()
plot(t*1e9, v-v_background)
plot(t*1e9, v)
.
plot(t_background*1e9, v_background)

print(max(v-v_background))
#%%

