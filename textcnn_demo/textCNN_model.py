import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from textcnn_demo.config import *
from textcnn_demo.tfrecord_generator import read_tf_records


def build_model(num_vocab, num_class, max_len, num_filters, filter_sizes, dropout_keep_prob, embedding_size):
    input_tensor = layers.Input(shape=(max_len,), dtype=tf.int32, name="inputs")
    embedding = layers.Embedding(input_dim=num_vocab, output_dim=embedding_size, input_length=max_len)(input_tensor)
    pooled_outputs = []
    for index, filter_size in enumerate(filter_sizes):
        conld = layers.Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, activation=tf.nn.leaky_relu,
                              padding="VALID")(embedding)
        pool_1d = layers.MaxPool1D(pool_size=max_len - filter_size + 1, padding="VALID")(conld)
        pooled_outputs.append(pool_1d)
    pool = tf.concat(values=pooled_outputs, axis=-1)
    pool_flat = tf.reshape(pool, [-1, len(filter_sizes) * num_filters])
    dropout = layers.Dropout(rate=1 - dropout_keep_prob)(pool_flat)
    score = layers.Dense(num_class, name="score")(dropout)
    prob = layers.Activation(activation="softmax", name="prob")(score)
    model = keras.Model(input_tensor, [score, prob])
    return model


def train():
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=1, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=1, mode='auto')
    ]
    inputs = read_tf_records(file_path, max_len, num_class)
    model = build_model(num_vocab, num_class, max_len, num_filters, filter_sizes, dropout_keep_prob, embedding_size)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss={'score': keras.losses.CategoricalCrossentropy(from_logits=True), "prob": None})
    history = model.fit(inputs, epochs=num_epoch, callbacks=callbacks)
    model.save(model_path, save_format="tf")
