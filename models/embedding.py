from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import jax.numpy as jnp
import jax

os.environ["JAX_PLATFORM_NAME"] = "gpu"
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)


# word = "Elephant"
# sentence = "I am a sentence for which I would like to get its embedding."
# paragraph = (
#     "Universal Sentence Encoder embeddings also support short paragraphs. "
#     "There is no hard limit on how long the paragraph is. Roughly, the longer "
#     "the more 'diluted' the embedding will be.")
# messages = [word, sentence, paragraph]

prompt = "Pick up the red cube."
messages = [prompt]

# Reduce logging output.
logging.set_verbosity(logging.ERROR)

message_embeddings = embed(messages)
jax_embeddings = jnp.array(message_embeddings)

# Save embeddings
# jax.save('embeddings/prompt_embeddings.npy', jax_embeddings)

# np.save('embeddings/prompt_embeddings.npy', jax_embeddings)
loaded_embeddings = np.load('embeddings/prompt_embeddings.npy')
repeated_embeddings = np.tile(loaded_embeddings, (15, 1))
import pdb; pdb.set_trace()

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
  print("Message: {}".format(messages[i]))
  print("Embedding size: {}".format(len(message_embedding)))
  message_embedding_snippet = ", ".join(
      (str(x) for x in message_embedding[:3]))
  print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
