import visa
from itertools import combinations
import pyfirmata
import time


class Digital_attenuator_31_75dB(object):
    
    def __init__(self, visa_name):
        self.rm = visa.ResourceManager()
        self.pyvisa = self.rm.open_resource(visa_name)
        self.pyvisa.timeout = 5000 # Set response timeout (in milliseconds)
        # self.pyvisa.query_delay = 1 # Set extra delay time between write and read commands
        global board
        board = pyfirmata.Arduino(visa_name)
    def read(self):
        return self.pyvisa.read()
    
    def write(self, string):
        self.pyvisa.write(string)

    def query(self, string):
        return self.pyvisa.query(string)

    def close(self):
        self.pyvisa.close()
        
    def reset(self):
        self.write('*RST')

    def identify(self):
        return self.query('*IDN?')

# g = GenericInstrument('GPIB0::24')
# g.identify()
    
    #Current pin set-up to linduino:
    #digitalPin2 = orange = LE always keep this HIGH
    #digitalPin3 = yellow = 1dB
    #digitalPin4 = green = 0.5 dB
    #digitalPin5 = blue = 0.25 dB
    #digitalPin6 = purple = 16dB
    #digitalPin7 = grey = 4dB
    #digitalPin8 = white = 8dB
    #digitalPin9 = black = 2dB
    #board.digital[10].write(0)
    #time.sleep(1)
    #print(board.digital[10].read())
    
    def set_attenuation(self, attenuation):
        
        if attenuation % 0.25 != 0:
            print('attenuation must be in intervals of 0.25 dB!')
        else:
             #function that returns list of db gates to turn on
            def SumTheList(thelist, target):
                arr = []
                p = []    
                if len(thelist) > 0:
                    for r in range(0,len(thelist)+1):        
                        arr += list(combinations(thelist, r))
            
                    for item in arr:        
                        if sum(item) == target:
                            p.append(item)
                return p
            
            
            #funtion that returns ordered bit list (0.25,0.5,1,2,4,8,16) ie 10 => (2,8) => [0,0,0,1,0,1,0] 
            def bit_list_order(sum_list):
                db_list = [0.25,0.5,1,2,4,8,16]
                bit_list = []
                for db in db_list:
                    bit = 0
                    for j in range(len(sum_list[0])):
                        if db == sum_list[0][j]:
                            bit = 1
                    bit_list.append(bit)
                return bit_list
            
            #find pin values 1 or o depending on attenuation
            db_list = [0.25,0.5,1,2,4,8,16]
            bit_ordered = bit_list_order(SumTheList(db_list, attenuation))      
            pin_0_25db = bit_ordered[0]
            pin_0_50db = bit_ordered[1]
            pin_1db = bit_ordered[2]
            pin_2db = bit_ordered[3]
            pin_4db = bit_ordered[4]
            pin_8db = bit_ordered[5]
            pin_16db = bit_ordered[6]
            
            time.sleep(0.5)
            board.digital[2].write(1) #set Latch Enabled HIGH
            board.digital[3].write(pin_1db)
            board.digital[4].write(pin_0_50db)
            board.digital[5].write(pin_0_25db)
            board.digital[6].write(pin_16db)
            board.digital[7].write(pin_4db)
            board.digital[8].write(pin_8db)
            board.digital[9].write(pin_2db)
    


#%%

digital_attenuator = Digital_attenuator_31_75dB('COM5')  
digital_attenuator.set_attenuation(20)

#%%
# Close all open resources

rm = visa.ResourceManager()
[i.close() for i in rm.list_opened_resources()]

    
