
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from classifier import Classify #User defined module


def main():
	baseDir = '/Users/abhisheksingh29895/Dropbox/AdvancedML_Project_/Data/' # Base directory
	params = {'max_depth':8,
			'subsample':0.5,
			'verbose':2,
			'random_state':0,
			'min_samples_split':20,
			'min_samples_leaf':20,
			'max_features':30,
			'n_estimators': 12000,
			'learning_rate': 0.002}
	clf = GradientBoostingClassifier(**params) #Fitting a model with the above parameters

	# Generate a submission with corr32 and all metrics
	test = Classify(trainFile=baseDir+'workspace/trainMetrics.csv',
					orderFile=baseDir+'moby/corr32.csv')
	test.testAndOutput(clf=clf, testFile=baseDir+'workspace/testMetrics.csv',
		orderFile=baseDir+'moby/testCorr32.csv', outfile='submit32.sub')

	# Generate a submission with corr64 and no time metrics
	noTime = np.array(range(150) + range(385,448))
	test = Classify(trainFile=baseDir+'workspace/trainMetrics.csv',
					orderFile=baseDir+'moby/corr64.csv', useCols=noTime)
	test.testAndOutput(clf=clf, testFile=baseDir+'workspace/testMetrics.csv',
		orderFile=baseDir+'moby/testCorr64.csv', outfile='submit64.sub')

	# Blend (Half + Half)
	s32 = np.loadtxt('submit32.sub',delimiter=',')
	s64 = np.loadtxt('submit64.sub',delimiter=',')
	sub_ = 0.5*s32 + 0.5*s64
	np.savetxt('blend.sub',sub_,delimiter=',')



#Calling the main function()
if __name__=="__main__":
	main()