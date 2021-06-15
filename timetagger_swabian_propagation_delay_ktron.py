from amcc.instruments.rigol_dg5000 import RigolDG5000
from amcc.instruments.lecroy_620zi import LeCroy620Zi
from amcc.instruments.switchino import Switchino
from amcc.instruments.tektronix_awg610 import TektronixAWG610
from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.agilent_53131a import Agilent53131a

import datetime
import TimeTagger
from TimeTagger import setLogger, createTimeTagger, Combiner, Coincidence, Counter, Countrate
from TimeTagger import Correlation, TimeDifferences, TimeTagStream, Scope, Event, CHANNEL_UNUSED, UNKNOWN, LOW, HIGH, LOGGER_WARNING
import pickle
import time
import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt

#Set-up instruments
awgpulse = TektronixAWG610('GPIB0::1')
awgsin = TektronixAWG610('GPIB0::23') # Sine generator
counter = Agilent53131a('GPIB0::10::INSTR')
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')

# create a timetagger instance
tagger = createTimeTagger()
try:
    tagger.reset()
except:
    pass
time.sleep(1)

#Runtime warning ignore;

#%%
#Set bias/ pulse
##Setting up sin wave (from ADAMs' code)
sin_bias_period = 10e-3 # Period of sine wave, in seconds
num_pulses_per_period = 1

# Setup heater-pulse AWGpulse
num_samples_delay = 511
num_samples_write = 1
marker_data =  [0] + [1]*num_samples_write + [0]*(num_samples_delay-1)
voltage_data = [-1] + [1]*num_samples_write + [-1]*(num_samples_delay-1)
marker_data = marker_data*num_pulses_per_period
voltage_data = voltage_data*num_pulses_per_period
awgpulse.create_waveform(voltages = voltage_data, filename = 'temp.wfm', clock = None, marker1_data = marker_data, auto_fix_sample_length = False)
awgpulse.load_file('temp.wfm')
awgpulse.set_vhighlow(vlow = 0, vhigh = 1)
awgpulse.set_marker_vhighlow(vlow = 0, vhigh = 1)


#awgw.set_trigger_mode(continuous_mode=True)
awgpulse.set_trigger_mode(trigger_mode=True)
awgpulse.set_output(True)

# Setup sine-bias AWG
awgsin.set_mode(fg_mode = True)
awgsin.set_lowpass_filter(20e6, channel = 1)
awgsin.set_trigger_mode(continuous_mode=True)
awgsin.set_clock(freq = 1/sin_bias_period) # Waveform is 1000 samples long, so 1000/1e5 = 10 ms period
awgsin.set_vhighlow(vlow = 0, vhigh = 0.08/2) # Inputting to resistor, so set to 1/2 value
awgsin.set_output(True, channel = 1)


pulse_rate = 1/sin_bias_period*num_pulses_per_period # Number of input pulses per second


#%% Get correlation measurement
#Setup
#GOAL: MAKE FUNCTIONS TO SCAN IBIAS AND VP. 1ST MUST GET TOLERANCE RIGHT!

#Parameters
time_to_measure = 1

trigger_level1 = 0.5
trigger_level2 = 0.500
binwidth_ps = 1
n_bins = 100000



# GOOD VALUES
# Negative channel numbers indicated "falling" edge
tagger.setTriggerLevel(1, trigger_level1)
tagger.setTriggerLevel(2, trigger_level2)

dead_time_ps = 2000
# Negative channel numbers indicated "falling" edge
tagger.setDeadtime(1, dead_time_ps)
tagger.setDeadtime(2, dead_time_ps)
time_in_ps = time_to_measure*1e12

# https://www.swabianinstruments.com/static/documentation/TimeTagger/api/Measurements.html?highlight=getdata#correlation

#%%PARAMETERS
#Actual values
sample = 'A19A17'
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + sample + 'time_trigger'
save_path = 'C:/Users/dsr1/se062/time_delay_measurements/'

vpp_sin = [0.2,0.4]
vpp_pulse = np.linspace(0.2,2,100)

dB = 20

Rb = 10e3
t_p = 1e-9
awgpulse.set_clock(1/t_p)

#%%FUNCTIONS

#Returns stats of each run ie for each vp and Ib
def statistics(x_axis, y_histogram): 
    x_axis_nonzero = []
    y_histogram_nonzero = []
    data_tot = []
    
    for i in range(len(x_axis)):
        
        if y_histogram[i] > 0:
            
            x_axis_nonzero.append(x_axis[i])
            y_histogram_nonzero.append(y_histogram[i])
            
        for i in range(len(x_axis_nonzero)):
                
            for j in range(y_histogram_nonzero[i]):
                data_tot.append(x_axis_nonzero[i])
        if not data_tot:
            mean = np.nan
            median = np.nan
            stdv = np.nan
        else:
            mean = np.mean(data_tot)/1000
            median = np.median(data_tot)/1000
            stdv = np.std(data_tot)/1000
    return mean, median, stdv


def find_histogram_median(x_axis, y_histogram):
    if np.sum(y_histogram)==0:
        return np.nan
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]
    
    
    ch = np.cumsum(y_histogram)
    idx, val = find_nearest(ch, np.max(ch)/2)
    return x_axis[idx]



#set ibias and v pulse, for each pair take measurement and return average, median, stdv with previous function
def time_delay_values(vpp_sin,vpp_pulse):
  
    data_list = []
    
    #First set ibias and vpulse
    for v1 in vpp_sin:
        
        awgsin.set_vpp(v1/2)            #set ibias 
        awgsin.set_voffset(v1/4)
        awgpulse.set_voffset(0)    #set pulse
        
        
        for v2 in vpp_pulse:
            
            ib = (v1/Rb)*1e6  #append params to list
            vp = v2/(10**(dB/20))
            vp_actual = v2
            power = ((v2/(10**(dB/20)))**2)/50
            rb = Rb
            db = dB
           
            
            awgpulse.set_vpp(v2)    #set pulse
            
            time.sleep(.5)
            #Take the data
            correlation = TimeTagger.Correlation(tagger, channel_1=1, channel_2=2, binwidth=binwidth_ps, n_bins=n_bins) 
            x_axis = correlation.getIndex()
            correlation.startFor(int(time_in_ps), clear=True)
            time.sleep(time_in_ps/1e12 + 0.1)
            y_histogram = correlation.getData()
            t_median = abs(find_histogram_median(x_axis, y_histogram))
            
            data = dict(
                ib = ib,
                vp = vp,
                vp_actual = vp_actual,
                power = power,
                rb = rb,
                db = db,
                t_median = t_median,
                )
            data_list.append(data)
            
            print("vpp_sin = %0.2f / vpp_pulse = %0.2f / median = %s ps" % (v1, v2, t_median))

            
    df = pd.DataFrame(data_list)
    plt.scatter(df['power'], df['t_median'])
    plt.xlabel()
    plt.ylabel()
    df.to_csv(save_path + filename + '.csv')
    plt.savefig(save_path + filename, dpi = 300)
    
    return df




#%%

df = time_delay_values(vpp_sin, vpp_pulse)
#%% Runnin one value at a time without using function
data_list = []
v1 = 0.4
for v2 in vpp_pulse:
            
    ib = (v1/Rb)*1e6  #append params to list
    vp = v2/(10**(dB/20))
    vp_actual = v2
    power = ((v2/(10**(dB/20)))**2)/50
    rb = Rb
    db = dB
          
           
    awgpulse.set_vpp(v2)    #set pulse
           
    time.sleep(.5)
            #Take the data
    correlation = TimeTagger.Correlation(tagger, channel_1=1, channel_2=2, binwidth=binwidth_ps, n_bins=n_bins) 
    x_axis = correlation.getIndex()
    correlation.startFor(int(time_in_ps), clear=True)
    time.sleep(time_in_ps/1e12 + 0.1)
    y_histogram = correlation.getData()
    t_median = abs(find_histogram_median(x_axis, y_histogram))
            
    data = dict(
        ib = ib,
        vp = vp,
        vp_actual = vp_actual,
        power = power,
        rb = rb,
        db = db,
        t_median = t_median,
        )
    data_list.append(data)
            
    print("vpp_sin = %0.2f / vpp_pulse = %0.2f / median = %s ps" % (v1, v2, t_median))

    
df = pd.DataFrame(data_list)
plt.scatter(df['power'], df['t_median'])
plt.xlabel('power (W)')
plt.ylabel('t_delay (ps)')
plt.title('time_delay ' + str(trigger_level2)+ 'V trigger')
df.to_csv(save_path + filename + '.csv')
plt.savefig(save_path + filename, dpi = 300)




#%% Lecroy commands for screenshot/ plots
channels = []
labels = []
channels.append('C2'); labels.append('Marker reference')
channels.append('C3'); labels.append('Crosstalk response')
#channels.append('C3'); labels.append('')
#channels.append('C4'); labels.append('LED pulse sync clock')
# channels.append('F1'); labels.append('SNSPD crosstalk average')
#channels.append('F2'); labels.append('')
#channels.append('M1'); labels.append('')
#channels.append('M2'); labels.append('')
#channels.append('M3'); labels.append('')
lecroy.save_screenshot('test.png') #screenshot
plot_channels = True
tscale = 1e9
tscale_label = 'Time (ns)'
vscale = 1e3
vscale_label = 'Voltage (mV)'

#%%Graphing the timetrigger data

data = pd.read_csv(r'C:/Users/dsr1/se062/time_delay_measurements/2021-06-15 12-40-33A19A17_bias.csv')

data_10 = data[data['ib'] == 10]
data_20 = data[data['ib'] == 20]
data_30 = data[data['ib'] == 30]
data_40 = data[data['ib'] == 40]

plt.scatter(data_10['power'], data_10['t_delay-t_overbiased'], label = '10 uA')
plt.scatter(data_20['power'], data_20['t_delay-t_overbiased'], label = '20 uA')
plt.scatter(data_30['power'], data_30['t_delay-t_overbiased'], label = '30 uA')
plt.scatter(data_40['power'], data_40['t_delay-t_overbiased'], label = '40 uA')
plt.legend()
plt.xlabel('power (W)')
plt.ylabel('t_median (ps)')

#%% Try over biasing and taking the measurement
t_p = 1e-9
awgpulse.set_clock(1/t_p)
correlation = TimeTagger.Correlation(tagger, channel_1=1, channel_2=2, binwidth=binwidth_ps, n_bins=n_bins) 
x_axis = correlation.getIndex()
correlation.startFor(int(time_in_ps), clear=True)
time.sleep(time_in_ps/1e12 + 1)
y_histogram = correlation.getData()
t_median = abs(find_histogram_median(x_axis, y_histogram))
print(t_median)
