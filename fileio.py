
""" fileio.py: File manager
"""
import sys
import random
import numpy as np
import aifc
from matplotlib import mlab
import cv2
import pylab as pl

#AIFF file to numpy arry
def ReadAIFF(file):
	s = aifc.open(file,'r')
	nFrames = s.getnframes()
	strSig = s.readframes(nFrames)
	return np.fromstring(strSig, np.short).byteswap()


#H0: Incorrect Whale clips, #H1: Correct whale clips
class TrainData(object):
	def __init__(self, fileName='', dataDir=''):
		""""""
		self.fileName = fileName
		self.dataDir = dataDir
		self.h1 = []
		self.h0 = []
		self.Load()

	def Load(self):
		""" Read the csv file and populate lists """
		file = open(self.fileName, 'r')
		self.hdr = file.readline().split('\n')[0].split(',')

		for line in file.readlines():
			tokens = line.split('\n')[0].split(',')
			if int(tokens[1]) == 0:
				self.h0.append(tokens[0])
			else:
				self.h1.append(tokens[0])
		file.close()
		self.numH1 = len(self.h1)
		self.numH0 = len(self.h0)

#return the spectogram & freq/time bins
	def H1Sample(self, index=None, params=None):
		if index == None:
			index = random.randint(0,self.numH1-1)
			print index
		s = ReadAIFF(self.dataDir+self.h1[index])
		P, freqs, bins = mlab.specgram(s, **params)
		return P, freqs, bins

	def H0Sample(self, index=None, params=None):
		if index == None:
			index = random.randint(0,self.numH0-1)
		s = ReadAIFF(self.dataDir+self.h0[index])
		P, freqs, bins = mlab.specgram(s, **params)
		return P, freqs, bins



#Point the test files to this
class TestData(object):
	def __init__(self, dataDir=''):
		self.dataDir = dataDir
		self.test = range(1,30000)
		self.nTest = 30000

	def TestSample(self, index=None, params=None):
		if index == None:
			index = random.randint(1,self.nTest)
		s = ReadAIFF(self.dataDir+'test'+('%i'%index)+'.aiff')
		P, freqs, bins = mlab.specgram(s, **params)
		return P, freqs, bins

