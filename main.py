'''
python watch_file.py -p1 python main.py -d `pwd` 
'''
from functions import generate_training_data
from genetic import genetic_fit
from nlp import (functions_from_prompt,
                 SAMPLE_PROMPT,
                )

'''
This generates training data, extracts functions, and runs the GP.  We see that
the last function printed out tends to be a sinusoid.
'''

if __name__=='__main__':

    trainingData=generate_training_data()
    otherPrimitives=functions_from_prompt(SAMPLE_PROMPT)
    pop,log,hof=genetic_fit(trainingData,
                            otherPrimitives=otherPrimitives)
 
    print([str(p) for p in pop[-5:]])
