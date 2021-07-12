#%%
import sys
if sys.version_info <= (3, 7):
    ValueError('You really should use Python 3.7 or greater with this script')

from amcc.instruments.lecroy_620zi import LeCroy620Zi
from amcc.instruments.tektronix_awg610 import TektronixAWG610
from amcc.instruments.agilent_53131a import Agilent53131a
# from amcc.instruments.srs_sim970 import SIM970
# from amcc.instruments.srs_sim928 import SIM928
# from amcc.instruments.switchino import Switchino

import datetime
import TimeTagger
from TimeTagger import createTimeTagger
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import itertools



#Returns stats of each run ie for each vp and Ib
def find_mean_std(x_axis, y_histogram): 
    # From https://stackoverflow.com/a/57400289
    probs = y_histogram / np.sum(y_histogram)
    mids = x_axis
    mean = np.sum(probs * mids)  
    sd = np.sqrt(np.sum(probs * (mids - mean)**2))
    return sd

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]
    
def find_histogram_median(x_axis, y_histogram):
    if np.sum(y_histogram)==0:
        return np.nan
    
    ch = np.cumsum(y_histogram)
    idx, val = find_nearest(ch, np.max(ch)/2)
    return x_axis[idx]

def parameter_combinations(parameters_dict):
    for k,v in parameters_dict.items():
        try: v[0]
        except: parameters_dict[k] = [v]
        if type(v) is str:
            parameters_dict[k] = [v]
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]


def reset_2x_awg_pulse_ktron_experiment(
    pulse_rate = 100,
):
    awgsin.reset()
    awgpulse.reset()
    time.sleep(0.1)
    sin_bias_period = 1/pulse_rate # Period of sine wave, in seconds
    num_pulses_per_period = 1

    # Setup heater-pulse AWGpulse
    num_samples_delay = 511
    num_samples_write = 1
    marker_data =  [0] + [1]*num_samples_write + [0]*(num_samples_delay-1)
    voltage_data = [-1] + [1]*num_samples_write + [-1]*(num_samples_delay-1)
    marker_data = marker_data*num_pulses_per_period
    voltage_data = voltage_data*num_pulses_per_period
    awgpulse.create_waveform(voltages = voltage_data, filename = 'temp.wfm', clock = None,
                            marker1_data = marker_data, auto_fix_sample_length = False)
    awgpulse.load_file('temp.wfm')
    awgpulse.set_vhighlow(vlow = 0, vhigh = 0.1)
    awgpulse.set_marker_vhighlow(vlow = 0, vhigh = 1)
    awgpulse.set_lowpass_filter(None, channel = 1)

    #awgw.set_trigger_mode(continuous_mode=True)
    awgpulse.set_trigger_mode(trigger_mode=True)
    awgpulse.set_output(True)

    # Setup sine-bias AWG
    awgsin.set_mode(fg_mode = True)
    awgsin.set_lowpass_filter(freq = 20e6, channel = 1)
    awgsin.set_function_waveform(waveform = 'sinusoid', channel = 1)
    awgsin.set_phase(90, channel = 1)
    awgsin.set_trigger_mode(continuous_mode=True)
    awgsin.set_clock(freq = 1/sin_bias_period) # Waveform is 1000 samples long, so 1000/1e5 = 10 ms period
    awgsin.set_vhighlow(vlow = 0, vhigh = 0.1) # Inputting to resistor, so set to 1/2 value
    awgsin.set_output(True, channel = 1)



def experiment_propagation_delay_timetagger(
    t = 2e-9,
    vbias = 0.5,
    rbias = 100e3,
    vp = 0.1,
    att_db = 40,
    count_time = 0.1,
    vp_splitter = True,
    tagger_ch1_trigger = 0.5,
    tagger_ch2_trigger = 0.25,
    tagger_dead_time = 20000,
    tagger_binwidth_ps = 1,
    tagger_n_bins = 100000,
    **kwargs,
    ):


    # Compute necessary parameters
    ibias = vbias/rbias
    vp_into_cryostat = vp*10**(-att_db/20)
    if vp_splitter is True:
        vp_into_cryostat = vp_into_cryostat/2
    power = (vp_into_cryostat**2)/50
    
    # Setup pulse-AWG parameters
    awgpulse.set_clock(1/t)
    awgpulse.set_vpp(abs(vp))
    awgpulse.set_voffset(vp/2)

    # Setup sine-wave-bias-AWG parameters
    awgsin.set_vpp(abs(vbias/2))
    awgsin.set_voffset(vbias/4)

    # Setup time tagger
    tagger_ch1 = 1
    tagger_ch2 = 2
    tagger.setTriggerLevel(tagger_ch1, tagger_ch1_trigger)
    tagger.setTriggerLevel(tagger_ch2, tagger_ch2_trigger)
    tagger.setDeadtime(tagger_ch1, tagger_dead_time)
    tagger.setDeadtime(tagger_ch2, tagger_dead_time)
    correlation = TimeTagger.Correlation(tagger, channel_1=tagger_ch1, channel_2=tagger_ch2,
                                    binwidth=tagger_binwidth_ps, n_bins=tagger_n_bins) 
    
    # Pause briefly to make sure all settings are entered
    time.sleep(100e-3)

    # Take the data
    correlation.startFor(int(count_time*1e12), clear=True)
    time.sleep(count_time + 0.05)
    y_histogram = correlation.getData()
    x_axis = correlation.getIndex()
    t_median = abs(find_histogram_median(x_axis, y_histogram))
    t_std = abs(find_mean_std(x_axis, y_histogram))
    
    # Store the data in a dictionary
    data = dict(
        vbias = vbias,
        rbias = rbias,
        ibias = ibias,
        vp = vp,
        att_db = att_db,
        vp_into_cryostat = vp_into_cryostat,
        power = power,
        t_median = t_median,
        t_std = t_std,
        vp_splitter = vp_splitter,
        tagger_ch1_trigger = tagger_ch1_trigger,
        tagger_ch2_trigger = tagger_ch2_trigger,
        tagger_dead_time = tagger_dead_time,
        tagger_binwidth_ps = tagger_binwidth_ps,
        tagger_n_bins = tagger_n_bins,
        **kwargs,
        )
    
    return data

#%% Initialize all the intruments

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

reset_2x_awg_pulse_ktron_experiment(pulse_rate = 100)

#%%

# Setup parameter sweeps -- any variable that is a list will be swept
# Lowest variable changes the fastest
parameter_dict = dict(
    device_name = "A17A19",
    vbias = [0.1,0.2],
    rbias = 10e3,
    vp = np.linspace(0.2,2,41),
    t = 2e-9,
    att_db = 0,
    count_time = 0.2,
    vp_splitter = True,
    tagger_ch1_trigger = None,
    tagger_ch2_trigger = 0.05,
    tagger_dead_time = 200000,
    tagger_binwidth_ps = 1,
    tagger_n_bins = 100000,
)


# Create combinations and manipulate them as needed
parameter_dict_list = parameter_combinations(parameter_dict)
for p_d in parameter_dict_list:
    p_d['tagger_ch1_trigger'] = p_d['vp']/4


# Run each parameter set as a separate experiment
data_list = []
for p_d in tqdm(parameter_dict_list):
    data_list.append(experiment_propagation_delay_timetagger(**p_d))


# Save data
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S propagation_delay_timetagger ') + testname
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')

#%%
for name, gd in df.groupby('ibias'):
    plt.plot(gd.ibias,gd.t_median, marker = '.', label = name*1e6)
plt.legend()
# df.groupby('ibias').plot('vp','t_median', marker = '.', ax=plt.gca())
# df.plot('vp','t_std', marker = '.')


#%%


# #%%

# # https://www.swabianinstruments.com/static/documentation/TimeTagger/api/Measurements.html

# time_in_ps = count_time*1e12

# #PARAMETERS
# #Actual values
# sample = 'A20A26'
# filename = sample + 'time_trigger'
# save_path = 'C:/Users/dsr1/se062/time_delay_measurements/'

# vpp_sin =  [0.1,0.2,0.3,0.4]
# vpp_pulse = np.linspace(0.2,2,50)

# dB = 10

# Rb = 10e3
# t_p = 1e-9
# awgpulse.set_clock(1/t_p)

# #%%FUNCTIONS




# #set ibias and v pulse, for each pair take measurement and return average, median, stdv with previous function
# def time_delay_values_marker(vpp_sin,vpp_pulse):
  
#     data_list = []
    
#     #First set ibias and vpulse
#     for v1 in vpp_sin:
        
#         awgsin.set_vpp(v1/2)            #set ibias 
#         awgsin.set_voffset(v1/4)
#         awgpulse.set_voffset(0)    #set pulse
        
#         for v2 in vpp_pulse:
            
#             ib = (v1/Rb)  #append params to list
#             vp = v2/(10**(dB/20))
#             vp_actual = v2
#             power = ((v2/(10**(dB/20)))**2)/50
#             rb = Rb
#             db = dB
           
            
#             awgpulse.set_vpp(v2)    #set pulse
            
#             time.sleep(.5)
#             #Take the data
#             correlation = TimeTagger.Correlation(tagger, channel_1=1, channel_2=2, binwidth=binwidth_ps, n_bins=n_bins) 
#             x_axis = correlation.getIndex()
#             correlation.startFor(int(time_in_ps), clear=True)
#             time.sleep(time_in_ps/1e12 + 0.1)
#             y_histogram = correlation.getData()
#             t_median = abs(find_histogram_median(x_axis, y_histogram))
#             t_std = abs(find_mean_std(x_axis, y_histogram))
            
#             data = dict(
#                 ib = ib,
#                 vp = vp,
#                 vp_actual = vp_actual,
#                 power = power,
#                 rb = rb,
#                 db = db,
#                 t_median = t_median,
#                 t_std = t_std
#                 )
#             data_list.append(data)
            
#             print("vpp_sin = %0.2f / vpp_pulse = %0.2f / median = %s ps" % (v1, v2, t_median))

           
#     df = pd.DataFrame(data_list)
#     plt.scatter(df['power'], df['t_median'], label = '%0.1f uA'%ib)
#     plt.xlabel('power (W)')
#     plt.ylabel('t_delay (ps)')
#     plt.legend()
#     plt.title('t_delay')
#     df.to_csv( save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + filename + '.csv')
#     plt.savefig( save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename, dpi = 300)
    
#     return df

# #for initiating tagger with the heater pulse split in 2
# def time_delay_values_heater_pulse(vpp_sin,vpp_pulse):
    
#     tagger.setTriggerLevel(2, v_trig2) #initially set trigger 2 for nanowire ouput pulse
#     data_list = []
#     #First set ibias and vpulse
#     for v1 in vpp_sin:
        
#         awgsin.set_vpp(v1/2)       #set ibias 
#         awgsin.set_voffset(v1/4)
#         awgpulse.set_voffset(0)    #set pulse offset
        
#         for v2 in vpp_pulse:
            
#             v_trig1 = v2/4
#             tagger.setTriggerLevel(1, v_trig1)
            
            
#             ib = (v1/Rb) #append params to list
#             v_half = v2/2
#             vp = v_half/(10**(dB/20))
#             vp_actual = v_half
#             power = ((v_half/(10**(dB/20)))**2)/50
#             rb = Rb
#             db = dB
           
#             awgpulse.set_vpp(v2)    #set pulse
#             time.sleep(.1)
#             #Take the data
#             correlation = TimeTagger.Correlation(tagger, channel_1=1, channel_2=2, binwidth=binwidth_ps, n_bins=n_bins) 
#             x_axis = correlation.getIndex()
#             correlation.startFor(int(time_in_ps), clear=True)
#             time.sleep(time_in_ps/1e12 + 0.1)
#             y_histogram = correlation.getData()
#             t_median = abs(find_histogram_median(x_axis, y_histogram))
#             t_std = abs(find_mean_std(x_axis, y_histogram))
            
#             data = dict(
#                 ib = ib,
#                 vp = vp,
#                 vp_actual = vp_actual,
#                 power = power,
#                 rb = rb,
#                 db = db,
#                 t_median = t_median,
#                 t_std = t_std,
#                 v_trig1 = v_trig1,
#                 v_trig2 = v_trig2,
#                 )
#             data_list.append(data)
            
#             print("vpp_sin = %0.2f / vpp_pulse = %0.2f / median = %s ps / std = %0.2f" % (v1, v2, t_median, t_std))

           
#     df = pd.DataFrame(data_list)
#     plt.scatter(df['power'], df['t_median'], label = '%0.1f uA'%ib)
#     plt.xlabel('power (W)')
#     plt.ylabel('t_delay (ps)')
#     plt.legend()
#     plt.title('t_delay')
#     df.to_csv( save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + filename + '.csv')
#     plt.savefig( save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename, dpi = 300)
    
#     return df
# #%%

# df = time_delay_values_marker(vpp_sin, vpp_pulse)
# #%%Try negative bias
# v = 1
# awgsin.set_vpp(v/2)
# awgsin.set_voffset(-v/4)
# #%% Runnin one value at a time without using function
# data_list = []
# v1 = 0.2
# awgsin.set_vpp(v1/2)
# awgsin.set_voffset(-v1/4)
# awgpulse.set_voffset(0)     
# tagger.setTriggerLevel(2,0.05)



# for v2 in vpp_pulse:
            
#     ib = (v1/Rb)  #append params to list
#     v_half = v2/2
#     vp = v_half/(10**(dB/20))
#     vp_actual = v_half
#     power = ((v_half/(10**(dB/20)))**2)/50
#     rb = Rb
#     db = dB
          
    
#     v_trig = vp/4
#     # v_trig = 0.05
#     tagger.setTriggerLevel(1,v_trig )
           
#     awgpulse.set_vpp(v2)    #set pulse
#     time.sleep(.1)
#             #Take the data
#     correlation = TimeTagger.Correlation(tagger, channel_1=1, channel_2=2, binwidth=binwidth_ps, n_bins=n_bins) 
#     x_axis = correlation.getIndex()
#     correlation.startFor(int(time_in_ps), clear=True)
#     time.sleep(time_in_ps/1e12 + 0.1)
#     y_histogram = correlation.getData()
#     t_median = abs(find_histogram_median(x_axis, y_histogram))
#     t_std = abs(find_mean_std(x_axis, y_histogram) )
            
#     data = dict(
#         ib = ib,
#         vp = vp,
#         vp_actual = vp_actual,
#         power = power,
#         rb = rb,
#         db = db,
#         t_median = t_median,
#         v_trig = v_trig,
#         t_std = t_std,
#         )
#     data_list.append(data)
            
#     print("vpp_sin = %0.2f / vpp_pulse = %0.2f / median = %s ps" % (v1, v2, t_median))


# df = pd.DataFrame(data_list)

# #df = pd.read_csv(r"C:\Users\dsr1\se062\time_delay_measurements\2021-06-30 14-05-14A20A26time_trigger.csv")
# plt.scatter(df['power'], df['t_median'])
# plt.xlabel('power (W)')
# plt.ylabel('t_delay (ps)')
# plt.title('time_delay ' + str(trigger_level2)+ 'V trigger' + ' -20 uA')
# df.to_csv( save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename + '.csv')
# plt.savefig(save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename, dpi = 300)


# #%%Scan of heater marker to heater pulse measurements for rrange of trigger lvls

# data_list = []
# trigger_lvl = np.linspace(.05,.5,50)
# v1 = 2
# awgsin.set_vpp(v1/2)
# awgsin.set_voffset(v1/4)
# vpp_pulse = np.linspace(0.2,2,20)

# for v in trigger_lvl:
    
#     tagger.setTriggerLevel(2, v)
#     tagger.setTriggerLevel(1,v)
    
#     for v2 in vpp_pulse:
                
#         #ib = (v1/Rb)*1e6  #append params to list
#         v_half = v2/2
#         vp = v_half/(10**(dB/20))
#         vp_actual = v_half
#         power = ((v_half/(10**(dB/20)))**2)/50
#         rb = Rb
#         db = dB
          
           
#         awgpulse.set_vpp(v2)    #set pulse
               
#         time.sleep(.1)
#                 #Take the data
#         correlation = TimeTagger.Correlation(tagger, channel_1=1, channel_2=2, binwidth=binwidth_ps, n_bins=n_bins) 
#         x_axis = correlation.getIndex()
#         correlation.startFor(int(time_in_ps), clear=True)
#         time.sleep(time_in_ps/1e12 + 0.1)
#         y_histogram = correlation.getData()
#         t_median = abs(find_histogram_median(x_axis, y_histogram))
#         #t_std = abs(find_mean_std(x_axis, y_histogram) )
            
#         data = dict(
#             #ib = ib,
#             vp = vp,
#             vp_actual = vp_actual,
#             power = power,
#             #rb = rb,
#             db = db,
#             t_median = t_median,
#             #t_std = t_std
#             v_trigger = v
#             )
#         data_list.append(data)
            
#         print("v_trig = %0.2f / vpp_pulse = %0.2f / median = %s ps" % (v, v2, t_median))

# df = pd.DataFrame(data_list)
# plt.scatter(df['power'], df['t_median'], c = df['v_trigger'], rasterized = True, vmin = 0, vmax = 0.5 )
# plt.colorbar()
# plt.xlabel('power (W)')
# plt.ylabel('t_delay (ps)')
# plt.title('Heater delay per vpulse and trigger lvl')
# df.to_csv( save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename + '.csv')
# plt.savefig(save_path + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +filename, dpi = 300)


# #%% Lecroy commands for screenshot/ plots
# channels = []
# labels = []
# channels.append('C2'); labels.append('Marker reference')
# channels.append('C3'); labels.append('Crosstalk response')
# #channels.append('P3'); labels.append('t_delay')
# #channels.append('C4'); labels.append('LED pulse sync clock')
# #channels.append('F3'); labels.append('t_delay')
# #channels.append('F2'); labels.append('')
# #channels.append('M1'); labels.append('')
# #channels.append('M2'); labels.append('')
# #channels.append('M3'); labels.append('')
# lecroy.save_screenshot(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + 'test.png') #screenshot
# plot_channels = True
# tscale = 1e9
# tscale_label = 'Time (ns)'
# vscale = 1e3
# vscale_label = 'Voltage (mV)'

# data = {}
# if plot_channels is True:
#     fig = plt.figure()
# for n,ch in enumerate(channels):
#     t,v = lecroy.get_wf_data(channel = ch)
#     data[ch + '_t'] = t
#     data[ch + '_v'] = v
#     if plot_channels is True:
        
#         plt.plot(t*tscale, v*1e3, label = ch + '-' + labels[n])
#         plt.ylabel(vscale_label)
#         plt.xlabel(tscale_label)
#         plt.legend()

