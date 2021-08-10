"""
@author: M. Hossein Eybposh, Summer 2019.
"""

from ctypes import *
import ctypes
import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread
import os

class SLM(object):
    '''
    Class for communication and control of the Spatial Light modulator.
    Inputs:
        dll_path   string, the path to the DLL library.
        lut_path   string, the path to the .LUT file in case you have a specific look up tabel.
        bit_depth   int, specifies the number bits that is defined for each pixel.
        is_nematic_type   int, refer to the SLM manual
        RAM_write_enable   int, refer to the SLM manual
        use_GPU   int, refer to the SLM manual
        max_transients   int, refer to the SLM manual
        wait_For_Trigger   int, behaves like a boolean, determines whether the hologram should be loaded immidiately or wait for a trigger.
        external_Pulse   int, refer to the SLM manual
        timeout_ms   int, write timeout in ms
        verbose   print timings or not
    Returns:
        an instance of the module.
    '''
    # Initialize the SLM
    def __init__(self,
                 dll_path = r"C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\Blink_C_wrapper.dll",
                 lut_path = r"C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\linear.LUT",
                 bit_depth = 12,
                 is_nematic_type = 1,
                 RAM_write_enable = 1,
                 use_GPU = 0,
                 max_transients = 10,
                 wait_For_Trigger = 0,
                 external_Pulse = 0,
                 timeout_ms = 5000,
                 verbose = False):
        num_boards_found = 0
        constructed_okay = 0
        
        self.dll_path = dll_path
        self.lut_path = lut_path
        self.bit_depth = c_int(bit_depth)
        self.num_boards_found = pointer(c_uint(num_boards_found))
        self.constructed_okay = pointer(c_int(constructed_okay))
        self.is_nematic_type = c_int(is_nematic_type)
        self.RAM_write_enable = c_int(RAM_write_enable)
        self.use_GPU = c_int(use_GPU)
        self.max_transients = c_int(max_transients)
        self.wait_For_Trigger = c_int(wait_For_Trigger)
        self.external_Pulse = c_int(external_Pulse)
        self.timeout_ms = c_int(timeout_ms)
        self.reg_lut = c_wchar_p('')
        self.dll = WinDLL(dll_path)
        
        self.slm_queue = Queue(maxsize=10)
        
        self.stop = False
        
        self.verbose = verbose
        
    def initialize(self, board_number = 1):
        '''
        This method creates the SDK that is later used to control the SLM.
        Inputs:
            board_number   int, specify which board you intend to access.
            
        '''
        print("Creating the SDK...")
        out = self.dll.Create_SDK(self.bit_depth,
                                      self.num_boards_found,
                                      self.constructed_okay,
                                      self.is_nematic_type,
                                      self.RAM_write_enable,
                                      self.use_GPU,
                                      self.max_transients,
                                      self.reg_lut)
        if out != -1:
            print("SDK creation Failed. Error number is: ", out)
        else:
            print("Successfully created the SDK. {} boards detected.".format(self.num_boards_found.contents.value))

        #% Load a LUT 
        self.board_number = board_number
        print("LUT file loaded to SLM number {}. LUT: ".format(board_number))
        print(self.lut_path)
        self.dll.Load_LUT_file(board_number, self.lut_path)
        
        self.height = self.dll.Get_image_height(board_number)
        self.width = self.dll.Get_image_width(board_number)
        print('dims are: ',[self.height,self.width])
        print("The dimensions are: "+str(self.height)+' X '+str(self.width))
        
        print("Starting the thread")
        # Define and start the thread that is responsible for prediction
        self.write_thread = Thread(target=self.write_from_queue, daemon=True)
        self.write_thread.start()
        self.counter = 0
        self.counter_q = 0
    
    def write_from_queue(self):
        while True:
            img = self.slm_queue.get()
            
            if img.shape[1]!=self.height or img.shape[2]!=self.width:
                print("Input size is not correct. ",img.shape)
            else:
                ImageOne = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                t0 = time.time()
                self.dll.Write_image(self.board_number, ImageOne, self.width*self.height, self.wait_For_Trigger, self.external_Pulse, self.timeout_ms)
                self.dll.ImageWriteComplete(self.board_number, self.timeout_ms)
                self.counter_q += 1
                t0 = time.time() - t0
                
                if self.verbose:
                    print("Write complete in {}ms".format(t0))
                
                if self.stop:
                    break
    
    def write(self, img):
        '''
        Write an image into the SLM.
        Inputs:
            img   two dimensional image to be written
        '''
        self.counter += 1
        self.slm_queue.put(img)
        
    def terminate(self):
        '''
        Close the SDK and the connection to the SLM. It is Extremely important
        that you close the SDK because oterwise you won't be able to reconnnect
        to the SLM and you have to restart both the SLM and therefore the computer.
        '''
        self.dll.Delete_SDK()
        self.stop = True