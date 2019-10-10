from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()
tfds.disable_progress_bar()

### Embedding layers
embedding_layer = layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1,2,3]))
result.numpy()

result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
result.shape

## Load data
# (train_data, test_data), info = tfds.load(
#     'imdb_reviews/subwords8k', 
#     split = (tfds.Split.TRAIN, tfds.Split.TEST), 
#     with_info=True, as_supervised=True)

actions = pd.read_csv('data/raw/actions.csv')
actions_long = np.array(actions.iloc[:,0:5])
outcome = np.array(actions['outcome'])
# sample_weights = np.array(actions['weight'])
sample_weights = np.ones_like(outcome)
X_train, X_test, y_train, y_test, sample_weights_train, sample_weights_test = \
  train_test_split(actions_long, outcome, sample_weights, test_size=0.33, random_state=42)
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights_train))
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test, sample_weights_test))

train_batches = train_data.shuffle(1000).batch(10)
test_batches = test_data.shuffle(1000).batch(10)
train_batch, train_label, sample_weights_train_temp = next(iter(train_batches))
sample_weights_train_temp.numpy()


## Sequential Model
embedding_dim=16
vocab_size = np.max(actions_long) + 1

model = keras.Sequential([
  layers.Embedding(vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(1, activation='sigmoid')
])

model.summary()


## Fit the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)


## Visualize

import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0,1))
plt.show()

## Output for Embedding Projector
e = model.layers[0]
weights = e.get_weights()[0]
weights2 = weights[np.unique(actions_long)]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io

out_v = io.open('vecs_noweight.tsv', 'w', encoding='utf-8')
out_m = io.open('meta_noweight.tsv', 'w', encoding='utf-8')

for num in np.arange(weights2.shape[0]):
  vec = weights2[num]
  out_m.write(str(np.unique(actions_long)[num]) + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()