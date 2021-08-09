# -*- coding: utf-8 -*-
"""
Ross Rucho
Socket programming class
07/08/19

"""

import socket
import numpy as np
from PIL import Image
#%%
# This class outlines a caiman network object
class NelNet(object):
    '''
    Object properties
    '''
    __serverSocket=None
    __host=None
    __port=None
    __clientSocket=None
    __clientAddress=None  
    __chunks=None
    __imageSize=None
    __imageDataType=None
    __imageDimensions=None
    __fix_msg=None
    
    '''
    Constructor
    '''
    # Initialize the server and connect to the client
    # 12000 is an arbitrary unreserved port number (non-privileged ports are > 1023)
    def __init__(self, host="", port=12000, fix_msg=2048):
        # Create socket with attributes IPv4 address family and TCP protocol
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Bind socket to the given host and port number
        serverSocket.bind((host, port))
        
        # Initiate server socket listen mode
        serverSocket.listen()
        
        # Assign class properties
        self.__serverSocket=serverSocket
        self.__host=host
        self.__port=port
        self.__fix_msg=fix_msg
        
        # Print acknowledgement to console
        print('The server is listening')
        
        # Accept an incoming connection
        clientSocket, clientAddress = serverSocket.accept()
    
        # Handshake and finish connection initialization
        print('Connected by', clientAddress)
        imageSize = clientSocket.recv(self.__fix_msg)
        clientSocket.sendall(imageSize)
        
        # Assign class properties
        self.__clientSocket=clientSocket
        self.__clientAddress=clientAddress
        self.__imageSize=int(imageSize.decode())
        self.__imageDataType='uint16'
        self.__imageDimensions=(512, 512)
        self.__chunks=bytearray(self.__imageSize)
    
    
    '''
    Class functions
    '''
    # Returns the server socket object
    def getServer(self):
        return self.__serverSocket
    
    # Returns the client socket object
    def getClient(self):
        return self.__clientSocket
    
    # Returns the address of the client socket
    def getClientAddress(self):
        return self.__clientAddress
    
    # Returns the size of the images being transmitted
    def getImageSize(self):
        return self.__imageSize
    
    # Returns the data type of the images being transmitted
    def getImageDataType(self):
        return self.__imageDataType
    
    # Returns the dimensions of the images being transmitted
    def getImageDimensions(self):
        return self.__imageDimensions
    
    # Returns the IP address of the host
    def getHost(self):
        return self.__host
    
    # Returns the port number of the host
    def getPort(self):
        return self.__port
    
    # This function allows for an entire image to be received
    def recvall(self):
        # Receives data from the socket until the entire image has been received
        bytes_recvd = 0          
        while bytes_recvd < self.__imageSize:
            chunk = self.__clientSocket.recv(min(self.__imageSize - bytes_recvd, self.__fix_msg))
            self.__chunks[bytes_recvd:bytes_recvd+len(chunk)] = chunk
            bytes_recvd += len(chunk)

    # Close all open sockets
    def closeAll(self):
        self.__clientSocket.close()
        self.__serverSocket.close()
        
    # This function allows for a single image to be acquired
    def startSingle(self):
        # Acquire one complete image as a byte stream
        self.recvall()
        
        # Convert the byte stream to a numpy array
        image = np.frombuffer(self.__chunks, dtype=self.__imageDataType).reshape(self.__imageDimensions)
        
        # Display the acquired image
        #Image.fromarray(image).show()
        
        # Return the image
        return image
    
    # This function allows for the socket to operate in batch mode
    def startBatch(self, batchSize=200):
        # Initialize a three dimensional array for images
        batch = np.zeros((batchSize, self.__imageDimensions[0], self.__imageDimensions[1]), dtype=self.__imageDataType)
        
        # Receive images until all of the images in a batch have been received
        s = "hSI.startFocus()".encode('utf-8')
        self.__clientSocket.send(s)
        count=0
        while(count < batchSize):
            self.recvall()
            image = np.frombuffer(self.__chunks, dtype=self.__imageDataType).reshape(self.__imageDimensions)
            batch[count]=image
            count = count+1
        s = "hSI.abort()".encode('utf-8')
        self.__clientSocket.send(s)
        
        # Return the image batch
        return batch
    
    def Focus(self):
        s = "hSI.startFocus()".encode('utf-8')
        self.__clientSocket.send(s)
    
    def Abort(self):
        s = "hSI.abort()".encode('utf-8')
        self.__clientSocket.send(s)
            

