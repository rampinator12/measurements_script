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
import math

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
awgpulse.set_lowpass_filter(9.9e+37, channel = 1)

#awgw.set_trigger_mode(continuous_mode=True)
awgpulse.set_trigger_mode(trigger_mode=True)
awgpulse.set_output(True)

# Setup sine-bias AWG
awgsin.set_mode(fg_mode = True)
awgsin.set_lowpass_filter(np.inf, channel = 1)
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

trigger_level1 = 0.5 #marker lvl trigger
trigger_level2 = 0.05
binwidth_ps = 1
n_bins = 100000



# GOOD VALUES
# Negative channel numbers indicated "falling" edge
tagger.setTriggerLevel(1, trigger_level1)
tagger.setTriggerLevel(2, trigger_level2)

dead_time_ps = 20000
# Negative channel numbers indicated "falling" edge
tagger.setDeadtime(1, dead_time_ps)
tagger.setDeadtime(2, dead_time_ps)
time_in_ps = time_to_measure*1e12

# https://www.swabianinstruments.com/static/documentation/TimeTagger/api/Measurements.html?highlight=getdata#correlation

#PARAMETERS
#Actual values
sample = 'A20A26'
filename = sample + 'time_trigger'
save_path = 'C:/Users/dsr1/se062/time_delay_measurements/'

vpp_sin =  [0.1,0.2,0.3,0.4]
vpp_pulse = np.linspace(0.2,2,50)

dB = 10

Rb = 10e3
t_p = 1e-9
awgpulse.set_clock(1/t_p)

#%%FUNCTIONS

#Returns stats of each run ie for each vp and Ib
def find_mean_std(x_axis, y_histogram): 
    # From https://stackoverflow.com/a/57400289
    probs = y_histogram / np.sum(y_histogram)
    mids = x_axis
    mean = np.sum(probs * mids)  
    sd = np.sqrt(np.sum(probs * (mids - mean)**2))
    return sd

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
        data_graph = []
        
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
            t_std = abs(find_mean_std(x_axis, y_histogram))
            
            data = dict(
                ib = ib,
                vp = vp,
                vp_actual = vp_actual,
                power = power,
                rb = rb,
                db = db,
                t_median = t_median,
                t_std = t_std
                )
            data_list.append(data)
            
            print("vpp_sin = %0.2f / vpp_pulse = %0.2f / median = %s ps" % (v1, v2, t_median))

           
    df = pd.DataFrame(data_list)
    plt.scatter(df['power'], df['t_median'], label = '%0.1f uA'%ib)
    plt.xlabel('power (W)')
    plt.ylabel('t_delay (ps)')
    plt.legend()
    plt.title('t_delay')
    df.to_csv( save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + filename + '.csv')
    plt.savefig( save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename, dpi = 300)
    
    return df




#%%

df = time_delay_values(vpp_sin, vpp_pulse)
#%%Try negative bias
v = 1
awgsin.set_vpp(v/2)
awgsin.set_voffset(-v/4)
#%% Runnin one value at a time without using function
data_list = []
v1 = 0.2
awgsin.set_vpp(v1/2)
awgsin.set_voffset(-v1/4)
awgpulse.set_voffset(0)     
tagger.setTriggerLevel(2,0.05)



for v2 in vpp_pulse:
            
    ib = (v1/Rb)  #append params to list
    v_half = v2/2
    vp = v_half/(10**(dB/20))
    vp_actual = v_half
    power = ((v_half/(10**(dB/20)))**2)/50
    rb = Rb
    db = dB
          
    
    v_trig = vp/4
    # v_trig = 0.05
    tagger.setTriggerLevel(1,v_trig )
           
    awgpulse.set_vpp(v2)    #set pulse
    time.sleep(.1)
            #Take the data
    correlation = TimeTagger.Correlation(tagger, channel_1=1, channel_2=2, binwidth=binwidth_ps, n_bins=n_bins) 
    x_axis = correlation.getIndex()
    correlation.startFor(int(time_in_ps), clear=True)
    time.sleep(time_in_ps/1e12 + 0.1)
    y_histogram = correlation.getData()
    t_median = abs(find_histogram_median(x_axis, y_histogram))
    t_std = abs(find_mean_std(x_axis, y_histogram) )
            
    data = dict(
        ib = ib,
        vp = vp,
        vp_actual = vp_actual,
        power = power,
        rb = rb,
        db = db,
        t_median = t_median,
        v_trig = v_trig,
        t_std = t_std,
        )
    data_list.append(data)
            
    print("vpp_sin = %0.2f / vpp_pulse = %0.2f / median = %s ps" % (v1, v2, t_median))


df = pd.DataFrame(data_list)

#df = pd.read_csv(r"C:\Users\dsr1\se062\time_delay_measurements\2021-06-30 14-05-14A20A26time_trigger.csv")
plt.scatter(df['power'], df['t_median'])
plt.xlabel('power (W)')
plt.ylabel('t_delay (ps)')
plt.title('time_delay ' + str(trigger_level2)+ 'V trigger' + ' -20 uA')
df.to_csv( save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename + '.csv')
plt.savefig(save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename, dpi = 300)


#%%Scan of heater marker to heater pulse measurements for rrange of trigger lvls

data_list = []
trigger_lvl = np.linspace(.05,.5,50)
v1 = 2
awgsin.set_vpp(v1/2)
awgsin.set_voffset(v1/4)
vpp_pulse = np.linspace(0.2,2,20)

for v in trigger_lvl:
    
    tagger.setTriggerLevel(2, v)
    tagger.setTriggerLevel(1,v)
    
    for v2 in vpp_pulse:
                
        #ib = (v1/Rb)*1e6  #append params to list
        v_half = v2/2
        vp = v_half/(10**(dB/20))
        vp_actual = v_half
        power = ((v_half/(10**(dB/20)))**2)/50
        rb = Rb
        db = dB
          
           
        awgpulse.set_vpp(v2)    #set pulse
               
        time.sleep(.1)
                #Take the data
        correlation = TimeTagger.Correlation(tagger, channel_1=1, channel_2=2, binwidth=binwidth_ps, n_bins=n_bins) 
        x_axis = correlation.getIndex()
        correlation.startFor(int(time_in_ps), clear=True)
        time.sleep(time_in_ps/1e12 + 0.1)
        y_histogram = correlation.getData()
        t_median = abs(find_histogram_median(x_axis, y_histogram))
        #t_std = abs(find_mean_std(x_axis, y_histogram) )
            
        data = dict(
            #ib = ib,
            vp = vp,
            vp_actual = vp_actual,
            power = power,
            #rb = rb,
            db = db,
            t_median = t_median,
            #t_std = t_std
            v_trigger = v
            )
        data_list.append(data)
            
        print("v_trig = %0.2f / vpp_pulse = %0.2f / median = %s ps" % (v, v2, t_median))

df = pd.DataFrame(data_list)
plt.scatter(df['power'], df['t_median'], c = df['v_trigger'], rasterized = True, vmin = 0, vmax = 0.5 )
plt.colorbar()
plt.xlabel('power (W)')
plt.ylabel('t_delay (ps)')
plt.title('Heater delay per vpulse and trigger lvl')
df.to_csv( save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename + '.csv')
plt.savefig(save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename, dpi = 300)


#%% Lecroy commands for screenshot/ plots
channels = []
labels = []
channels.append('C2'); labels.append('Marker reference')
channels.append('C3'); labels.append('Crosstalk response')
#channels.append('P3'); labels.append('t_delay')
#channels.append('C4'); labels.append('LED pulse sync clock')
#channels.append('F3'); labels.append('t_delay')
#channels.append('F2'); labels.append('')
#channels.append('M1'); labels.append('')
#channels.append('M2'); labels.append('')
#channels.append('M3'); labels.append('')
lecroy.save_screenshot(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + 'test.png') #screenshot
plot_channels = True
tscale = 1e9
tscale_label = 'Time (ns)'
vscale = 1e3
vscale_label = 'Voltage (mV)'

data = {}
if plot_channels is True:
    fig = plt.figure()
for n,ch in enumerate(channels):
    t,v = lecroy.get_wf_data(channel = ch)
    data[ch + '_t'] = t
    data[ch + '_v'] = v
    if plot_channels is True:
        
        plt.plot(t*tscale, v*1e3, label = ch + '-' + labels[n])
        plt.ylabel(vscale_label)
        plt.xlabel(tscale_label)
        plt.legend()

#%%Graphing the timetrigger data

data = pd.read_csv(r"C:\Users\dsr1\se062\time_delay_measurements\2021-06-16 14-51-24A20A26time_trigger.csv")

data_10 = data[data['ib'] == 10]
data_20 = data[data['ib'] == 20]
data_30 = data[data['ib'] == 30]
data_40 = data[data['ib'] == 40]

data_10['t_delay'] = data_10['t_median'] - 36610
data_20['t_delay'] = data_20['t_median'] - 36610
data_30['t_delay'] = data_30['t_median'] - 36610
data_40['t_delay'] = data_40['t_median'] - 36610
plt.scatter(data_10['power'], data_10['t_delay'], label = '10 uA')
plt.scatter(data_20['power'], data_20['t_delay'], label = '20 uA')
plt.scatter(data_30['power'], data_30['t_delay'], label = '30 uA')
plt.scatter(data_40['power'], data_40['t_delay'], label = '40 uA')
plt.legend()
plt.xlabel('power (W)')
plt.ylabel('t_delay (ps)')
plt.title('A11A25 time_delay' )
plt.savefig(save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + filename, dpi = 300)

#%% Experimental runs to check oscilloscope
awgsin.set_vpp(.1/2)
awgsin.set_voffset(.1/4)
awgpulse.set_vpp(0.1)
t_p = 1e-9
awgpulse.set_clock(1/t_p)
#%%
time.sleep(1)
correlation = TimeTagger.Correlation(tagger, channel_1=1, channel_2=2, binwidth=binwidth_ps, n_bins=n_bins) 
x_axis = correlation.getIndex()
correlation.startFor(int(time_in_ps), clear=True)
time.sleep(time_in_ps/1e12 + 1)
y_histogram = correlation.getData()
t_median = abs(find_histogram_median(x_axis, y_histogram))
print(t_median)
