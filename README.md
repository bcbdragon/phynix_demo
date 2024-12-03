# phynix_demo

Welcome to the Phynix Genetic Programming demonstration.  We will first go through installation and running instructions, and then describe the algorithm.  This will be accompanied by my LaTeX description of a pipeline to derive runnable code from scientific articles, and will relate to one phase of this - specifically, using GP to build runnable components of the pipeline.

# Installation

Please ensure you are using a Linux machine with about 8gb of memory.  I am running this on the latest version of Ubuntu.  Currently, this runs on Python 3.9.12, although it will probably work with later versions.

## OpenAPI key

You will need a key to OpenAI's API.  This can either be set as an environment variable, or you can supply it in a `.env` file.  In the same directory as your code, use a text editor to open `.env`, and input:
```
OPENAI_API_KEY='my_open_ai_api_key'
```
replacing `my_open_ai_api_key` with your key.

## Requirements

Now, install the requirements.  It is your choice whether you want to install them in a virtual environment or a docker.  We assume you are doing this in your normal environment:
```
pip install -r requirements.txt
```

# Running

Now, you are ready to run the code:
```
python main.py
```
You will receive output from the Genetic Programming at the end, looking like:
```
[... 'cosine_with_freq(mul(numpy_array(mul(x, -0.6821523338685462)), -0.6821523338685462), 0.007380134940533134)', 'cosine_with_freq(mul(x, x), 0.007380134940533134)']
```
These describe the functions used to fit a sinusoid in the training data.

# Description of the project

This is a first pass at the use of Genetic Programming (GP) to compose runnable functions.  The user, an engineer, has a set of training data, and they intuit that it may describe a wave.  If they do not think it is a wave, they would simply estimate the function with arithmetic primitives - the result can be seen if you run:
```
python genetic.py
```
In order to make their heuristic more accurate, they submit a prompt to the LLM, defined in `PRELUDE` and `SAMPLE_PROMPT` in `nlp.py`.  The program looks at two possible sources of new functions to approximate the training data.  The frist is parsed directly from the LLM result - basic numpy functions which are tested to ensure that they are scalar one-to-one.  The second is a tiny "database" consisting of "proprietary" functions, with descriptions.  In the second case, an embedding from the prompt is taken along with embeddings for the descriptions.  The database entry with the highest cosine similarity between the prompt and the description is then included as a candidate function.

Now, we have additional functions which we can include in the GP algorithm alongside the arithmetic primitives.  The fitness function for the GP is a mean-squared-error (MSE) - GP functions which perform better have lower MSE.  The terminal output of `main.py` describes a function which best approximates the training data.

One problematic feature is the fact that the arguments of the winning functon are often very arbitrary - for example `sine_with_freq(-0.0009989738372488954, protected_div(x, -0.10074518131868371))`.  This is the result of `ephemeral constants` - or random floats inserted into the parse tree.  Future versions of the algorithm should be able to optimize these values rather than depending on the unwieldy computations needed to approximate the actual constants used to generate the training data.

I do find it encouraging, however, that the algorithm gives us sinusoidal functions.

Please enjoy responsibly.