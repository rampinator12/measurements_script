# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:54:01 2021

@author: danar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\dsr1\se062\DSR_iv_heater_A19A17.csv')

df_1 = df[df['v_b'] == 0]
df_2 = df[(df['v_b'] > 0) & (df['v_b'] < 0.4)]
df_3 = df[(df['v_b'] > .5) & (df['v_b'] < 0.8)]
df_4 = df[(df['v_b'] > 1) & (df['v_b'] < 1.2)]
df_5 = df[(df['v_b'] > 1.4) & (df['v_b'] < 1.6)]
df_6 = df[(df['v_b'] > 1.8) & (df['v_b'] < 2)]
df_7 = df[(df['v_b'] > 2.2) & (df['v_b'] < 2.4)]
df_8 = df[(df['v_b'] > 2.6) & (df['v_b'] < 2.8)]
df_9 = df[(df['v_b'] > 3) & (df['v_b'] < 3.2)]
df_10 = df[(df['v_b'] > 3.3)]


           

df_1_ar = df_1.to_numpy()
df_2_ar = df_2.to_numpy()
df_3_ar = df_3.to_numpy()
df_4_ar = df_4.to_numpy()
df_5_ar = df_5.to_numpy()
df_6_ar = df_6.to_numpy()
df_7_ar = df_7.to_numpy()
df_8_ar = df_8.to_numpy()
df_9_ar = df_9.to_numpy()
df_10_ar = df_10.to_numpy()

print(df_1_ar)

#%% get max value for each heater current
print(df_10.max()['i_b'])
#%%
plt.plot(df_1_ar[:,5], df_1_ar[:,6], label = 'ih = -4.24e-8A')
plt.plot(df_2_ar[:,5], df_2_ar[:,6], label = 'ih = 3.81e-6A')
plt.plot(df_3_ar[:,5], df_3_ar[:,6], label = 'ih = 7.63e-6A')
plt.plot(df_4_ar[:,5], df_4_ar[:,6], label = 'ih = 1.14e-5A')
plt.plot(df_5_ar[:,5], df_5_ar[:,6], label = 'ih = 1.52e-5A')
plt.plot(df_6_ar[:,5], df_6_ar[:,6], label = 'ih = 1.90e-5A')
plt.plot(df_7_ar[:,5], df_7_ar[:,6], label = 'ih = 1.82e-6A')
plt.plot(df_8_ar[:,5], df_8_ar[:,6], label = 'ih = 1.85e-6A')
plt.plot(df_9_ar[:,5], df_9_ar[:,6], label = 'ih = 3.04e-5A')
plt.plot(df_10_ar[:,5], df_10_ar[:,6], label = 'ih = 3.42e-5A')
plt.xlabel('V(volts)')
plt.ylabel('i_b(A)')
plt.title("A19heater A17measured")
plt.legend()
plt.show()
