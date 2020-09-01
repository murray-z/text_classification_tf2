# -*- coding: utf-8 -*-
# https://keras.io/examples/nlp/text_classification_with_transformer/

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class MultiHeadsSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadsSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should divisible by number of heads = {num_heads}"
            )
        # 每个头的维度
        self.project_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value): # [batch_size, num_heads, seq_len, project_dim]
        # Q*V
        score = tf.matmul(query, key, transpose_b=True)  #[batch_size, num_heads, seq_len, seq_len]
        # dim of key
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        # Q*K/(sqrt(dim_key))
        scaled_score = score / tf.math.sqrt(dim_key)
        # softmax(Q*K/(sqrt(dim_key))
        weights = tf.nn.softmax(scaled_score, axis=-1)  #[batch_size, num_heads, seq_len, seq_len]
        # softmax(Q*K/(sqrt(dim_key)) * value
        output = tf.matmul(weights, value) #[batch_size, num_heads, seq_len, project_dim]
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.project_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # [batch_size, num_heads, seq_len, project_dim]


    def call(self, inputs):
        # inputs: [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        # query, key, val：[batch, seq_len, embed_dim]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        # query, key, val：[batch_size, seq_len, num_heads, project_dim]
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) #[batch_size, seq_len, num_heads, project_dim]
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim)) #[batch_size, seq_len, embed_dim]
        output = self.combine_heads(concat_attention) #[batch_size, seq_len, embed_size]
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # attention layer
        self.att = MultiHeadsSelfAttention(embed_dim, num_heads)
        # feed forward layer
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs+attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs, **kwargs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(inputs)
        return x+positions








