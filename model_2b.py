import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras import backend as K, regularizers, layers
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import (Input, Dense, Activation, Conv1D, Conv2D,
    BatchNormalization, Dropout, Add, Lambda, Reshape, Permute,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Multiply, Concatenate, MaxPooling1D)

class Channel_attention(Model):
    def __init__(self, num_ch):
        super(Channel_attention, self).__init__(name='')
        self.num_ch  = num_ch
        self.conv1   = Conv2D(filters=8, kernel_size=(1, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))
        self.conv2   = Conv2D(filters=8, kernel_size=(1, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))
        self.reshape = Reshape((num_ch, -1))
        self.gamma   = Variable(initial_value=0, dtype='float32', trainable=True)

    def get_config(self):
        config = super().get_config()
        config.update({"num_ch": self.num_ch})
        return config

    def call(self, x):
        proj_query = self.reshape(self.conv1(x))
        proj_key   = self.reshape(self.conv2(x))
        energy     = tf.matmul(proj_query, proj_key, transpose_b=True)
        max_H      = tf.math.reduce_max(energy, axis=2, keepdims=True)
        min_H      = tf.math.reduce_min(energy, axis=2, keepdims=True)
        energy     = (energy - min_H) / ((max_H - min_H) + 1e-8)
        attention  = tf.nn.softmax(energy)
        out = (self.gamma
               * tf.expand_dims(
                   tf.matmul(attention, tf.squeeze(x, axis=-1), transpose_a=True),
                   axis=3)
               + x)
        return out

class MyAttention(Model):
    def __init__(self, num_ch, num_t):
        super(MyAttention, self).__init__()
        self.num_ch = num_ch
        self.num_t  = num_t
        self.CA     = Channel_attention(num_ch=num_ch)

    def get_config(self):
        config = super().get_config()
        config.update({"num_ch": self.num_ch, "num_t": self.num_t})
        return config

    def call(self, x, training=None):
        return self.CA(x)

def TCN_block(input_layer, input_dimension, depth, kernel_size, filters,
              dropout, activation='relu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1,
                   activation='linear', padding='causal',
                   kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1,
                   activation='linear', padding='causal',
                   kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if input_dimension != filters:
        conv  = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1),
                       activation='linear', padding='causal',
                       kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1),
                       activation='linear', padding='causal',
                       kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out   = Activation(activation)(added)
    return out

def cbam_block_1d(input_feature, ratio=8, residual=False):
    cbam_feature = channel_attention_1d(input_feature, ratio)
    cbam_feature = spatial_attention_1d(cbam_feature)
    if residual:
        cbam_feature = Add()([input_feature, cbam_feature])
    return cbam_feature

def channel_attention_1d(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    reduced = max(1, channel // ratio)
    shared_layer_one = Dense(reduced, activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal',
                             use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling1D()(input_feature)
    avg_pool = Reshape((1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return Multiply()([input_feature, cbam_feature])

def spatial_attention_1d(input_feature):
    avg_pool     = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(input_feature)
    max_pool     = Lambda(lambda x: K.max(x, axis=2,  keepdims=True))(input_feature)
    concat       = Concatenate(axis=2)([avg_pool, max_pool])
    cbam_feature = Conv1D(filters=1, kernel_size=7, padding='same',
                          activation='sigmoid', kernel_initializer='he_normal',
                          use_bias=False)(concat)
    return Multiply()([input_feature, cbam_feature])

class GatedAttentionFusion(tf.keras.layers.Layer):
    def __init__(self, fused_dim=64):
        super().__init__()
        self.fused_dim  = fused_dim
        self.proj1      = tf.keras.layers.Dense(fused_dim)
        self.proj2      = tf.keras.layers.Dense(fused_dim)
        self.attn_dense = tf.keras.layers.Dense(fused_dim, activation='tanh')
        self.gate_score = tf.keras.layers.Dense(1)

    def get_config(self):
        config = super().get_config()
        config.update({"fused_dim": self.fused_dim})
        return config

    def call(self, x1, x2):
        p1      = self.proj1(x1)
        p2      = self.proj2(x2)
        stacked = tf.stack([p1, p2], axis=2)           # (B, T, 2, D)
        attn_h  = self.attn_dense(stacked)
        attn_s  = self.gate_score(attn_h)
        weights = tf.nn.softmax(attn_s, axis=2)
        fused   = tf.reduce_sum(weights * stacked, axis=2)
        return fused

def build_full(temporal_dimension=500, num_channels=3, num_classes=2):
    input_main  = Input(shape=(temporal_dimension, num_channels), name="input_main")
    input_theta = Input(shape=(temporal_dimension, num_channels), name="input_theta")
    input_alpha = Input(shape=(temporal_dimension, num_channels), name="input_alpha")
    input_beta  = Input(shape=(temporal_dimension, num_channels), name="input_beta")

    # --- Main Branch ---
    x_main = Reshape((num_channels, temporal_dimension, 1))(input_main)
    x_main = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_main)
    x      = Reshape((temporal_dimension, num_channels))(x_main)

    x1     = TCN_block(input_layer=input_main, input_dimension=num_channels, depth=2,
                       kernel_size=4, filters=16, dropout=0.3, activation='elu')

    x2     = tf.keras.layers.Concatenate()([x, x1])
    x3     = cbam_block_1d(x2)

    # --- Theta Branch ---
    x_theta = Reshape((num_channels, temporal_dimension, 1))(input_theta)
    x_theta = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_theta)
    xt      = Reshape((temporal_dimension, num_channels))(x_theta)

    # --- Alpha Branch ---
    x_alpha = Reshape((num_channels, temporal_dimension, 1))(input_alpha)
    x_alpha = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_alpha)
    xa      = Reshape((temporal_dimension, num_channels))(x_alpha)

    # --- Beta Branch ---
    x_beta = Reshape((num_channels, temporal_dimension, 1))(input_beta)
    x_beta = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_beta)
    xb     = Reshape((temporal_dimension, num_channels))(x_beta)

    # --- Sub-band Fusion ---
    x2_sub = tf.keras.layers.Concatenate()([xa, xt, xb])   # → (N, 500, 9)
    x31    = cbam_block_1d(x2_sub)

    # --- Gated Attention Fusion (main + sub-band) ---
    fused_dim = int(x3.shape[-1]) + int(x31.shape[-1])
    fused     = GatedAttentionFusion(fused_dim=fused_dim)(x3, x31)

    # --- Output Block ---
    out = Conv1D(32, 5, activation='relu', padding='same')(fused)
    out = MaxPooling1D(2)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.4)(out)

    out = Conv1D(64, 3, activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = GlobalAveragePooling1D()(out)
    out = Dropout(0.3)(out)

    out = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))(out)
    out = Dense(units=num_classes)(out)
    out = Activation('softmax')(out)

    model = Model(
        inputs=[input_main, input_theta, input_alpha, input_beta],
        outputs=out
    )
    return model
