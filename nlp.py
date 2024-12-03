'''
python watch_file.py -p1 python nlp.py -d `pwd` 
'''
from dotenv import load_dotenv
from openai import OpenAI
from functions import (sine_with_freq,
                       cosine_with_freq,
                       FUNCTION_DESCRIPTIONS,
                       FUNCTION_TUPLES,
                      )
import numpy as np

'''
This module handles LLM lookup tasks.
'''

#############
# CONSTANTS #
#############

# Please be sure to put your OPENAI_API_KEY in your .env file.

load_dotenv()

CLIENT=OpenAI()
EMBEDDING_MODEL='text-embedding-3-large'

# The minimum cosine similarity for a match.

SIM_THRESHOLD=0.35

# PROMPTS

# PRELUDE defines the task which the LLM will accomplish, including lookup of 
# basic numpy functions appropriate to the task.

PRELUDE='''
You are going to read a description of a system written by an engineer.  This engineer wants to write a mathematical description of the system.  You are not responsible for doing this, but what you are responsible for is listing the numpy functions which would be involved in this description.  You will list each function after a prompt 'FUNCTION:'.  Do not include the 'numpy' or 'np' prefix.

Here is the engineer's prompt: {prompt}  
'''

# SAMPLE_PROMPT deals with the specific problem - we are looking for an 
# analytical solution to a system of wwaves.

SAMPLE_PROMPT='''
I want to describe a system of waves, possibly one, possibly the other.  This system takes one argument and one argument only.
'''


#############
# FUNCTIONS #
#############

# OPENAI access

def embed_strings(strings,
                  model=EMBEDDING_MODEL,
                  client=CLIENT,
                 ):
    '''
    For a series of strings, put their embeddings into rows of numpy array.
    '''
    response=client.embeddings.create(model=model,input=strings)
    embeddings=np.array([d.embedding for d in response.data])
    
    return embeddings


def embeddings_to_function_tups(strings,
                                functionTuples,
                                model=EMBEDDING_MODEL,
                                client=CLIENT,
                               ):
    '''
    This constructs a matrix containing the embeddings of function descriptions,
    and then creates a closure to match a query string to one of these
    embeddings, returning the string, the function tuple, and the similarity.
    '''
    embeddings=embed_strings(strings)
    
    def search_function_tuples(query):

        queryEmbedding=embed_strings([query])[0,:]
        prod=np.dot(embeddings,queryEmbedding)
        maxIndex=np.argmax(prod)
        
        return strings[maxIndex],functionTuples[maxIndex],prod[maxIndex]

    
    return search_function_tuples
        
                                
def run_prompt(text,
               prelude=PRELUDE,
               client=CLIENT,
              ):
    '''
    Boilerplate to run an OpenAI prompt.  Gets the function names in lines,
    returns them.
    '''
    prompt=prelude.format(prompt=text)
    
    completion=client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {
                'role': 'system', 
                'content': 'You are an engineering assistant, helping solve mathematical problems.'},
            {
                'role': "user",
                'content': prompt
            }])

    response=completion.choices[0].message.content
    lines=[line.replace('FUNCTION:','').lstrip(' ').rstrip()
           for line in response.split('\n') if 'FUNCTION:' in line]
    
    return lines


# Converting responses to code

def is_float_or_int(val):
    '''
    This ensures that our value is a type that can be processed by a function
    in the GP.
    '''
    conditions=[isinstance(val,np.float64),
                isinstance(val,np.int64),
                isinstance(val,float),
                isinstance(val,int),
                not isinstance(val,np.ndarray),
                not isinstance(val,list),
                not isinstance(val,dict),
                not isinstance(val,tuple),
              ]

    return any(conditions)


def is_scalar_function(f):
    '''
    This ensures that the function is scalar and one-to-one.
    '''
    return isinstance(f,np.ufunc) and f.signature is None


def filter_lines(lines):
    '''
    Here, we look up our lines in numpy, check their signatures and parity,
    ensuring that they are scalar and one-to-one.  If they are, then we 
    add then to numpy_fs.
    '''
    numpy_fs=[]

    for line in lines:

        # We use a try-catch block to test the function.
        try:

            numpy_f=getattr(np,line) # Is it in numpy?
            val=numpy_f(0) # Does it work on a scalar value?

            if is_float_or_int(val): # Does it return a scalar value?

                functionName=f'numpy_{line}'

                # Let's put it in a closure to make its output a Python float.
                def float_numpy_f(x):

                    if is_scalar_function(numpy_f):

                        return float(numpy_f(x))
                    
                    return 1

                # Run it ...
                float_numpy_f(0)
                # Rename it
                float_numpy_f.__name__=functionName
                # Add it to our list.
                numpy_fs.append((float_numpy_f,1))

        except Exception as e:

            print(e)

    return numpy_fs


def lookup_lines(lines,search_f,simThreshold=SIM_THRESHOLD):
    '''
    We take the closure defined in search_f to retrieve the description and
    the function tuple.  We put it in the dictionary for uniqueness.
    '''
    searchTups=[search_f(line) for line in lines]
    uniqueFunctions={desc:f for (desc,f,sim) in searchTups 
                     if sim > simThreshold}
    
    return list(uniqueFunctions.values())


def convert_lines(lines,search_f):
    '''
    Here, we are looking up functions directly from the response and from
    our database, appending them.
    '''
    return filter_lines(lines)+lookup_lines(lines,search_f) 


def functions_from_prompt(text,
                          functionDescriptions=FUNCTION_DESCRIPTIONS,
                          functionTuples=FUNCTION_TUPLES,
                         ):
    '''
    This function builds a closure, runs the prompt, and retrieves functions
    from the response and the database.
    '''
    search_f=embeddings_to_function_tups(functionDescriptions,
                                         functionTuples,
                                        )
    lines=run_prompt(text)
    functions=convert_lines(lines,search_f)

    return functions


if __name__=='__main__':

    functions=functions_from_prompt(SAMPLE_PROMPT)

    print(functions)
