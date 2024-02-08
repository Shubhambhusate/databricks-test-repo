# Databricks notebook source
pip install huggingface_hub["tensorflow"]

# COMMAND ----------


dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras("keras-io/mobile-vit-xxs")

# COMMAND ----------

from tensorflow.keras.models import Model, save_model
import tensorflow as tf
import datetime
import random
import shutil
from huggingface_hub import from_pretrained_keras
now = datetime.datetime.now()
task_id = now.strftime("%y%m%d%H%M%S") + str(random.randint(100, 999))
path = "/Volumes/shubham_test/test_volume/volume_shubham"
tmp_path = rf"prefix_{task_id}.h5"
save_model(model, tmp_path)
print("test1")

save_file_path = rf"{path}/{task_id}.h5"
shutil.move(tmp_path, save_file_path)
print("test2")

# COMMAND ----------

# MAGIC %sh ls

# COMMAND ----------

# MAGIC %sh ls /Volumes/shubham_test/test_volume/volume_shubham
