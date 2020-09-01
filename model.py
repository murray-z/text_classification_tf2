# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras import layers
import os
from transformer_layer import *


# 设置显卡及按需增长
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


class TextCnn(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, class_num):
        super(TextCnn, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        self.conv1d_layer = layers.Conv1D(filters=128, kernel_size=5, activation='relu')
        self.max_pool_layer = layers.GlobalMaxPool1D()
        self.dense_layer = layers.Dense(units=64, activation='relu')
        self.output_layer = layers.Dense(units=class_num, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.embedding_layer(inputs)
        x = self.conv1d_layer(x)
        x = self.max_pool_layer(x)
        x = self.dense_layer(x)
        y = self.output_layer(x)
        return y


class TextMultiKernalCnn(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, class_num):
        super(TextMultiKernalCnn, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        self.conv_layer1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
        self.pool_layer1 = layers.GlobalMaxPool1D()
        self.conv_layer2 = layers.Conv1D(filters=32, kernel_size=4, activation='relu', padding='same')
        self.pool_layer2 = layers.GlobalMaxPool1D()
        self.conv_layer3 = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')
        self.pool_layer3 = layers.GlobalMaxPool1D()
        self.dense_layer = layers.Dense(units=64, activation='relu')
        self.output_layer = layers.Dense(class_num, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.embedding_layer(inputs)
        x1 = self.conv_layer1(x)
        x1 = self.pool_layer1(x1)
        x2 = self.conv_layer2(x)
        x2 = self.pool_layer2(x2)
        x3 = self.conv_layer3(x)
        x3 = self.pool_layer3(x3)
        x = tf.concat([x1, x2, x3], axis=1)
        x = self.dense_layer(x)
        y = self.output_layer(x)
        return y


class TextBiLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, class_num):
        super(TextBiLSTM, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        self.bilstm_layer = layers.Bidirectional(layers.LSTM(units=128, return_sequences=False))
        self.output_layer = layers.Dense(units=class_num, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.embedding_layer(inputs)
        x = self.bilstm_layer(x)
        y = self.output_layer(x)
        return y


class TextBiGRU(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, class_num):
        super(TextBiGRU, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        self.bigru_layer = layers.Bidirectional(layers.GRU(units=128, return_sequences=False))
        self.output_layer = layers.Dense(units=class_num, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.embedding_layer(inputs)
        x = self.bigru_layer(x)
        y = self.output_layer(x)
        return y


class TextCnnLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, class_num):
        super(TextCnnLSTM, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        self.conv_layer = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.pool_layer = layers.MaxPool1D()
        self.lstm_layer = layers.LSTM(units=100)
        self.output_layer = layers.Dense(units=class_num, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.embedding_layer(inputs)
        x = self.conv_layer(x)
        x = self.pool_layer(x)
        x = self.lstm_layer(x)
        y = self.output_layer(x)
        return y


class TextCnnGRU(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, class_num):
        super(TextCnnGRU, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        self.conv_layer = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.pool_layer = layers.MaxPool1D()
        self.lstm_layer = layers.GRU(units=100)
        self.output_layer = layers.Dense(units=class_num, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.embedding_layer(inputs)
        x = self.conv_layer(x)
        x = self.pool_layer(x)
        x = self.lstm_layer(x)
        y = self.output_layer(x)
        return y


class TextBilstmAttention(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, class_num):
        super(TextBilstmAttention, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        self.bilstm_layer = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))
        self.attention_layer = layers.Attention()
        self.pool_layer = layers.GlobalAveragePooling1D()
        self.concat_layer = layers.Concatenate()
        self.dense_layer = layers.Dense(units=64, activation='relu')
        self.output_layer = layers.Dense(units=class_num, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        q = self.embedding_layer(inputs)
        v = self.embedding_layer(inputs)
        q = self.bilstm_layer(q)
        v = self.bilstm_layer(v)
        q_v = self.attention_layer([q, v])
        q = self.pool_layer(q)
        q_v = self.pool_layer(q_v)
        x = self.concat_layer([q, q_v])
        x = self.dense_layer(x)
        y = self.output_layer(x)
        return y


class TextCnnAttention(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, class_num):
        super(TextCnnAttention, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embed_size)
        self.conv_layer = layers.Conv1D(filters=128, kernel_size=5, padding='same')
        self.attention_layer = layers.Attention()
        self.pool_layer = layers.GlobalAveragePooling1D()
        self.concat_layer = layers.Concatenate()
        self.dense_layer = layers.Dense(units=64, activation='relu')
        self.output_layer = layers.Dense(units=class_num, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        q = self.embedding_layer(inputs)
        v = self.embedding_layer(inputs)
        q = self.conv_layer(q)
        v = self.conv_layer(v)
        q_v = self.attention_layer([q,v])
        q = self.pool_layer(q)
        q_v = self.pool_layer(q_v)
        x = self.concat_layer([q, q_v])
        x = self.dense_layer(x)
        y = self.output_layer(x)
        return y


class Transformer(tf.keras.Model):
    def __init__(self, maxlen, vocab_size, embed_size, num_heads, ff_dim, class_num):
        super(Transformer, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen=maxlen, vocab_size=vocab_size,
                                                         embed_dim=embed_size)
        self.transformer_block = TransformerBlock(embed_size, num_heads, ff_dim)
        self.pooling = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation='rele')
        self.dense2 = layers.Dense(class_num, activation='softmax')
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, inputs, **kwargs):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.pooling(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        y = self.dense2(x)
        return y












