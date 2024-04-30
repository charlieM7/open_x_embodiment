# base code from https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder

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

# prompt = "Grasp the red cube then lift it in the air"
# prompt = "Pick up the red cube."
# prompt = "Open the door."
# prompt = "Grip the red block and raise it up off the table."
# prompt = "Grip the yellow door handle."
# prompt = "Move to the right"
prompt = "Open the door on the left"
messages = [prompt]

# Reduce logging output.
logging.set_verbosity(logging.ERROR)

message_embeddings = embed(messages)
jax_embeddings = jnp.array(message_embeddings)

np.save('embeddings/prompt_embeddings7.npy', jax_embeddings)
repeated_embeddings = np.tile(jax_embeddings, (15, 1))

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
  print("Message: {}".format(messages[i]))
  print("Embedding size: {}".format(len(message_embedding)))
  message_embedding_snippet = ", ".join(
      (str(x) for x in message_embedding[:3]))
  print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
