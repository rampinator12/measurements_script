#%%============================================================================
# Load functions
#==============================================================================

from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.switchino import Switchino
from amcc.instruments.srs_sim921 import SIM921
from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.jds_ha9 import JDSHA9
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import pyvisa as visa
import itertools



#%%============================================================================
# Setup instruments
#==============================================================================

# Close all open resources
rm = visa.ResourceManager()
[i.close() for i in rm.list_opened_resources()]

# Connect to instruments
counter = Agilent53131a('GPIB0::12::INSTR')
vs = SIM928('ASRL9::INSTR', 2)
dmm = SIM970('ASRL9::INSTR', 3)
att = JDSHA9('GPIB0::11::INSTR')
switch = Switchino('COM8')
#srs_temp_sensor = SIM921('GPIB0::6', sim900port = 5)


# Setup SRS voltage source and DMM
vs.reset()
vs.set_output(True)
# dmm.set_impedance(gigaohm = True, channel = 3)

# Setup counter
counter.set_impedance(ohms = 50)
counter.set_coupling(dc = True)
counter.setup_timed_count()
counter.set_100khz_filter(False)
counter.set_trigger(trigger_voltage = 0.1, slope_positive = True, channel = 1)



## Setup attenuator
att.set_wavelength(1550)
att.set_attenuation_db(0)
att.set_beam_block(False )

#%%============================================================================
#Functions
#==============================================================================
def parameter_combinations(parameters_dict): # mix params for different sweeps
    for k,v in parameters_dict.items():
        try: v[0]
        except: parameters_dict[k] = [v]
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]

def snspd_measurement(
        att_db,
        rbias,
        ibias,
        vtrig,
        delay,
        count_time,
        **kwargs,
        ):
    
    #Set-up experiment, initialize intruments
    vbias = ibias*rbias
    vs.set.voltage(vbias)
    att.set_attenuation_db(att_db)
    counter.set_trigger(trigger_voltage = v, slope_positive = True, channel = 1)
    time.sleep(delay) # wait for intruments to be ready ~ 250-500 ms
    
    #take measurement/ return data + paarams for each sweep
    counts = counter.timed_count(counting_time = count_time)
    count_rate = counts/count_time
    
    data = dict(
        att_db = att_db,
        rbias = rbias,
        ibias = ibias,
        vbias = vbias,
        vtrig = vtrig,
        delay = delay,
        count_time = count_time,
        counts = counts,
        count_rate = count_rate,
        **kwargs,
        )
    
    return data

    
    
#%%============================================================================
# RUN EXPERIMENT
#============================================================================== 
#zero out voltage
vs.set_voltage(0)

#Make combos
test_name = 'enter chip name/ die here'
device  = 'S3',
temperature = 1.7,

parameter_dict = dict(   #NOTE: lowest variable (order-wise) changes teh fastest)
    
    
    att_db = 0,
    rbias = 10e3,
    vtrig = np.linspace(0,0.2,100),
    ibias = [0,20e-6,24e-6,25e-6,27e-6],
    delay = 100e-3,
    count_time = 0.1,
    device = device,
    temperature = temperature,
    
    )

# Create combinations and manipulate them as needed
parameter_dict_list = parameter_combinations(parameter_dict)
data_list = []
for p_d in tqdm(parameter_dict_list):
    data_list.append(snspd_measurement(**p_d))


vs.set_voltage(0) # turn off voltage at the end of a run

#Save data 

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ktron min energy and delay') + test_name + device
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')

#%% Trigger lvl sweep

plt.figure()

for name, gd in df.groupby(['ibias']):
    plt.semilogy(gd.vtrig, gd.count_rate, label = 'ibias = %0.1f uA' %(name*1e6))
    plt.xlabel('vtrig (V)') 
    plt.ylabel('count rate (1/s)')   
    plt.legend()

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd counting')
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')
plt.savefig(filename + 'trig_lvl.png', dpi = 300)

#%%Counts vs Ibias

plt.figure()
plt.semilogy (df['ibias'], df['count_rate'], marker = '.')
plt.xlabel('ibias (uA)') 
plt.ylabel('counts rate (1/s)')   

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd counting')
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')
plt.savefig(filename + '.png', dpi = 300)

#%% Counts vs attenuation
plt.figure()
plt.semilogy(df['att_db'], df['count_rate'], marker = '.')
plt.xlabel('attenuation (dB)') 
plt.ylabel('count rate (1/s)')   

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd counting')
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')
plt.savefig(filename + '.png', dpi = 300)


#%%============================================================================
# Trigger Level Experiment
#==============================================================================
device  = 'S3'
att_db = 0
temperature = 1.7
rbias = 10e3
ibias = [0,20e-6,24e-6,25e-6,27e-6]
vtrig = np.linspace(0,0.2,100)
delay = 100e-3
count_time = 0.1

data_list = []


for i in ibias : #set ibias/ atenuation in this for loop
    
    vbias = i*rbias
    vs.set_voltage(vbias)
    att.set_attenuation_db(att_db)
    
    for v in vtrig: #set trigger, count and measure/ calculate values 
        
        counter.set_trigger(trigger_voltage = v, slope_positive = True, channel = 1)
        time.sleep(delay)
        counts = counter.timed_count(counting_time = count_time)
        
        count_rate = (counts/count_time)
        
        data = dict(
            device = device,
            rbias = rbias,
            ibias = i,
            vbias = vbias, # derived
            vtrig = v,
            counts = counts,
            count_time = count_time,
            count_rate = count_rate,  # derived
            att_db = att_db,
            delay = delay, # Delay after setting voltage/attenuation
            temperature = temperature, # Get from ppms? Or enter manually
            
            ) 
        
        data_list.append(data)

vs.set_voltage(0)
df = pd.DataFrame(data_list)   
plt.figure()

for name, gd in df.groupby(['ibias']):
    plt.semilogy(gd.vtrig, gd.count_rate, label = 'ibias = %0.1f uA' %(name*1e6))
    plt.xlabel('vtrig (V)') 
    plt.ylabel('count rate (1/s)')   
    plt.legend()

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd counting')
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')
plt.savefig(filename + '.png', dpi = 300)
#%%============================================================================
# Counts v Ibias
#==============================================================================

device = 'S3' 
att_db = 0
temperature = 1.7
rbias = 97e3
ibias = np.linspace(7.5e-6,9e-6,100)
vtrig = 50e-3
delay = 250e-3
count_time = 1

data_list = []

counter.set_trigger(trigger_voltage = vtrig, slope_positive = True, channel = 1) # set trigger before anything else
for i in ibias : #set ibias/ atenuation in this for loop
    
    vbias = i*rbias
    vs.set_voltage(vbias)
    time.sleep(delay)
    
    counts = counter.timed_count(counting_time = count_time)
    count_rate = (counts/count_time)
        
    data = dict(
        device = device,
        rbias = rbias,
        ibias = i,
        vbias = vbias, # derived
        vtrig = v,
        counts = counts,
        count_time = count_time,
        count_rate = count_rate,  # derived
        att_db = att_db,
        delay = delay, # Delay after setting voltage/attenuation
        temperature = temperature, # Get from ppms? Or enter manually
        ) 

    data_list.append(data)

vs.set_voltage(0)
df = pd.DataFrame(data_list)   
plt.figure()
plt.semilogy (df['ibias'], df['count_rate'], marker = '.')
plt.xlabel('ibias (uA)') 
plt.ylabel('counts rate (1/s)')   

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd counting')
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')
plt.savefig(filename + '.png', dpi = 300)

#%%============================================================================
# Attenuation vs counts
#==============================================================================

device = 'N5'
att_db = np.linspace(0,10,100)
temperature = 1.7
rbias = 10e3
ibias = 26e-6
vtrig = 50e-3
delay = 250e-3
count_time = 1

data_list = []

counter.set_trigger(trigger_voltage = vtrig, slope_positive = True, channel = 1) # set trigger before anything else
for a in att_db : #set ibias/ atenuation in this for loop
    
    vbias = ibias*rbias
    vs.set_voltage(vbias)
    att.set_attenuation_db(a)
    time.sleep(delay)
    
    counts = counter.timed_count(counting_time = count_time)
    count_rate = (counts/count_time)
        
    data = dict(
        device = device,
        rbias = rbias,
        ibias = ibias,
        vbias = vbias, # derived
        vtrig = v,
        counts = counts,
        count_time = count_time,
        count_rate = count_rate,  # derived
        att_db = a,
        delay = delay, # Delay after setting voltage/attenuation
        temperature = temperature, # Get from ppms? Or enter manually
        ) 
    data_list.append(data)
     
vs.set_voltage(0)
att.set_attenuation_db(0)     

df = pd.DataFrame(data_list)   
plt.figure()
plt.semilogy(df['att_db'], df['count_rate'], marker = '.')
plt.xlabel('attenuation (dB)') 
plt.ylabel('count rate (1/s)')   

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd counting')
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')
plt.savefig(filename + '.png', dpi = 300)

