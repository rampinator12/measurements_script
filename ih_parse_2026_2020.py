# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:35:35 2021

@author: dsr1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\dsr1\se062\DSR_iv_heater_A26A20.csv')

df_1 = df[df['v_b'] == 0]
df_2 = df[(df['v_b'] > 0) & (df['v_b'] < 0.7)]
df_3 = df[(df['v_b'] > .8) & (df['v_b'] < 1.3)]
df_4 = df[(df['v_b'] > 1.4) & (df['v_b'] < 1.9)]
df_5 = df[(df['v_b'] > 2) & (df['v_b'] < 2.5)]
df_6 = df[(df['v_b'] > 2.6) & (df['v_b'] < 3.1)]
df_7 = df[(df['v_b'] > 3.3) & (df['v_b'] < 3.7)]
df_8 = df[(df['v_b'] > 4) & (df['v_b'] < 4.3)]
df_9 = df[(df['v_b'] > 4.7) & (df['v_b'] < 4.9)]
df_10 = df[(df['v_b'] > 5)]


           

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

#%%
plt.plot(df_1_ar[:,5], df_1_ar[:,6], label = 'ih = -4.27e-8A')
plt.plot(df_2_ar[:,5], df_2_ar[:,6], label = 'ih = 6.07e-6A')
plt.plot(df_3_ar[:,5], df_3_ar[:,6], label = 'ih = 1.22e-5A')
plt.plot(df_4_ar[:,5], df_4_ar[:,6], label = 'ih = 1.83e-5A')
plt.plot(df_5_ar[:,5], df_5_ar[:,6], label = 'ih = 2.40e-6A')
plt.plot(df_6_ar[:,5], df_6_ar[:,6], label = 'ih = 2.42e-6A')
plt.plot(df_7_ar[:,5], df_7_ar[:,6], label = 'ih = 3.66e-5A')
plt.plot(df_8_ar[:,5], df_8_ar[:,6], label = 'ih = 4.27e-5A')
plt.plot(df_9_ar[:,5], df_9_ar[:,6], label = 'ih = 4.38e-5A')
plt.plot(df_10_ar[:,5], df_10_ar[:,6], label = 'ih = 4.76e-5A')
plt.xlabel('V(volts)')
plt.ylabel('i_b(A)')
plt.title("A26heater A20measured")
plt.legend()
plt.show()
