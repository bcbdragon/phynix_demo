'''
python watch_file.py -p1 python functions.py -d `pwd`
'''
import numpy as np

'''
This module contains several functions which will be used in our GP algorithm,
along with the data structures needed to access them with embeddings.
'''

#############
# CONSTANTS #
#############

FREQ_IN_HZ=0.01


#############
# FUNCTIONS #
#############

def protected_div(left, right):
     '''
     This is protected division, returning 0 if there is a divide-by-zero.
     '''
     try:
        
         return left / right 
    
     except ZeroDivisionError:
            
         return 1


def radians(t,f):
     '''
     This takes a time term t and a frequency term f and converts this into 
     radians, or 2*pi*f*t.
     '''
     return 2*np.pi*f*t


def sinusoid_with_freq(x,freq,sinusoid_f=np.sin):
     '''
     Generalized function for sinusoids, takes x and freq, converts into radians,
     applies sinusoid.
     '''
     return sinusoid_f(radians(x,freq))


def sine_with_freq(x,freq=FREQ_IN_HZ):
     '''
     Computes the sine with the proper signature.
     '''
     return sinusoid_with_freq(x,freq,np.sin)


def cosine_with_freq(x,freq=FREQ_IN_HZ):
     '''
     Computes the cosine with the proper signature.
     '''
     return sinusoid_with_freq(x,freq,np.cos)


def generate_training_data(start=0,end=100,freq=FREQ_IN_HZ):
     '''
     Generates training data for the GP.
     '''
     return [(x,sine_with_freq(x,freq)) for x in range(start,end)]


def mse_eval(func,trainingData):
     '''
     Evaluates the Mean Squared Error for the function on the training data.
     This will be used as a fitness function in the GP.
     '''
     lnTrainingData=len(trainingData)
     sqErrors=[(func(x) - y)**2 for (x,y) in trainingData]
     sumSqErrors=np.sum(sqErrors)

     return float(sumSqErrors/lnTrainingData)


##############################
# TUPLES FOR FUNCTION LOOKUP #
##############################

'''
These will serve as our 'database' when we do function lookup.  Notice that 
FUNCTION_TUPLES requires the function and its parity.  This is for GP.
'''
FUNCTION_DESCRIPTIONS=['a cosine with two arguments, one for time and the other for frequency',
                       'a sine with two arguments, one for time and the other for frequency',
                      ]
FUNCTION_TUPLES=[(cosine_with_freq,2),
                 (sine_with_freq,2),
                ]


if __name__=='__main__':

     trainingData=generate_training_data()
     mse=mse_eval(sine_with_freq,trainingData)

     print(mse)

