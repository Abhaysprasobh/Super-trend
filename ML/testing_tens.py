import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
print(tf.keras.__version__)
print(tf.__version__)
