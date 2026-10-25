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
    def __init__(self,num_ch):
        super(Channel_attention,self).__init__(name='')
        self.conv1      = Conv2D(filters=8,kernel_size=(1,1),kernel_constraint = max_norm(2., axis=(0,1,2)))
        self.conv2      = Conv2D(filters=8,kernel_size=(1,1),kernel_constraint = max_norm(2., axis=(0,1,2)))
        self.reshape    = Reshape((num_ch,-1))
        self.gamma      = Variable(initial_value=0,dtype='float32',trainable=True)

    def call(self,x):
        proj_query      = self.reshape(self.conv1(x))
        proj_key        = self.reshape(self.conv2(x))

        energy          = tf.matmul(proj_query,proj_key,transpose_b=True)

        max_H           = tf.math.reduce_max(energy,axis=2,keepdims=True)
        min_H           = tf.math.reduce_min(energy,axis=2,keepdims=True)

        tmp_b           = (energy-min_H)
        tmp_c           = (max_H-min_H)+1e-8
        energy          = tmp_b/tmp_c
        attention       = tf.nn.softmax(energy)

        out = self.gamma * tf.expand_dims(tf.matmul(attention,tf.squeeze(x,axis=-1),transpose_a=True),axis=3) + x

        return out

class MyAttention(Model):
    def __init__(self,num_ch,num_t):
        super(MyAttention,self).__init__()
        self.CA = Channel_attention(num_ch=num_ch)
        # self.TA = Time_attention(num_t=num_t)

    def call(self,x,training=None):
        out1 = self.CA(x)
        # out2 = self.TA(x)

        # out = tf.concat([out1, out2],axis=-1)

        return out1

def TCN_block(input_layer,input_dimension,depth,kernel_size,filters,dropout,activation='relu'):
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_dimension != filters):
        conv = Conv1D(filters,kernel_size=1,padding='same')(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)

    for i in range(depth-1):
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out

def cbam_block_1d(input_feature, ratio=8, residual=False):
    cbam_feature = channel_attention_1d(input_feature, ratio)
    cbam_feature = spatial_attention_1d(cbam_feature)

    if residual:
        cbam_feature = Add()([input_feature, cbam_feature])
    return cbam_feature

def channel_attention_1d(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

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
    avg_pool = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(input_feature)  # shape: (batch, time, 1)
    max_pool = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(input_feature)   # shape: (batch, time, 1)

    concat = Concatenate(axis=2)([avg_pool, max_pool])  # shape: (batch, time, 2)

    cbam_feature = Conv1D(filters=1,
                          kernel_size=7,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return Multiply()([input_feature, cbam_feature])

class GatedAttentionFusion5(tf.keras.layers.Layer):
    def __init__(self, fused_dim=64):
        super(GatedAttentionFusion5, self).__init__()
        self.fused_dim = fused_dim

        # 5 projection layers to project all inputs to fused_dim
        self.projs = [layers.Dense(fused_dim) for _ in range(5)]

        # attention scoring layer
        self.attn_dense = layers.Dense(fused_dim, activation='tanh')
        self.gate_score = layers.Dense(1)  # score scalar per source

    def call(self, x1, x2):
        # Step 1: Project all to same fused_dim
        projected = [proj(x) for proj, x in zip(self.projs, [x1, x2])]  # List of (B, T, D)

        # Step 2: Stack over source dimension: (B, T, 5, D)
        stacked = tf.stack(projected, axis=2)

        # Step 3: Compute attention weights per source (B, T, 5, 1)
        attn_hidden = self.attn_dense(stacked)          # (B, T, 5, D)
        attn_scores = self.gate_score(attn_hidden)      # (B, T, 5, 1)
        attn_weights = tf.nn.softmax(attn_scores, axis=2)  # softmax over source axis (5)

        # Step 4: Apply attention weights: (B, T, D)
        fused = tf.reduce_sum(attn_weights * stacked, axis=2)

        return fused

class GatedAttentionFusion(tf.keras.layers.Layer):
    def __init__(self, fused_dim=64):
        super().__init__()
        self.fused_dim = fused_dim

        # Projection layers for the two inputs
        self.proj1 = tf.keras.layers.Dense(fused_dim)
        self.proj2 = tf.keras.layers.Dense(fused_dim)

        # Attention layers
        self.attn_dense = tf.keras.layers.Dense(fused_dim, activation='tanh')
        self.gate_score = tf.keras.layers.Dense(1)

    def call(self, x1, x2):
        # Project inputs to same dimension
        p1 = self.proj1(x1)
        p2 = self.proj2(x2)

        # Stack along source dimension -> (B, T, 2, D)
        stacked = tf.stack([p1, p2], axis=2)

        # Attention scores
        attn_hidden = self.attn_dense(stacked)
        attn_scores = self.gate_score(attn_hidden)

        # Attention weights across sources
        attn_weights = tf.nn.softmax(attn_scores, axis=2)

        # Weighted fusion
        fused = tf.reduce_sum(attn_weights * stacked, axis=2)

        return fused

def build_additive(temporal_dimension=500, num_channels=22, num_classes=4):
    input_main  = Input(shape=(temporal_dimension, num_channels), name="input_main")
    input_theta = Input(shape=(temporal_dimension, num_channels), name="input_theta")
    input_alpha = Input(shape=(temporal_dimension, num_channels), name="input_alpha")
    input_beta  = Input(shape=(temporal_dimension, num_channels), name="input_beta")

    # --- Main Branch ---
    x_main = Reshape((num_channels, temporal_dimension, 1))(input_main)
    x_main = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_main)
    x      = Reshape((temporal_dimension, num_channels))(x_main)
    x1     = TCN_block(input_layer=input_main, input_dimension=16, depth=2,
                       kernel_size=4, filters=22, dropout=0.3, activation='elu')
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
    x2_sub = tf.keras.layers.Concatenate()([xa, xt, xb])
    x31    = cbam_block_1d(x2_sub)

    # ✅ KEY CHANGE: Project both branches to same dim, then Add
    proj_dim = 64
    x3_proj  = Conv1D(proj_dim, 1, padding='same')(x3)   # 1x1 conv to project
    x31_proj = Conv1D(proj_dim, 1, padding='same')(x31)
    fused    = tf.keras.layers.Add()([x3_proj, x31_proj])

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
    out = Dense(units=4)(out)
    out = Activation('softmax')(out)

    model_add_fusion = Model(inputs=[input_main, input_theta, input_alpha, input_beta], outputs=out)
    return model_add_fusion


def build_concatenation(temporal_dimension=500, num_channels=22, num_classes=4):
    input_main  = Input(shape=(temporal_dimension, num_channels), name="input_main")
    input_theta = Input(shape=(temporal_dimension, num_channels), name="input_theta")
    input_alpha = Input(shape=(temporal_dimension, num_channels), name="input_alpha")
    input_beta  = Input(shape=(temporal_dimension, num_channels), name="input_beta")

    # --- Main Branch ---
    x_main = Reshape((num_channels, temporal_dimension, 1))(input_main)
    x_main = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_main)
    x      = Reshape((temporal_dimension, num_channels))(x_main)
    x1     = TCN_block(input_layer=input_main, input_dimension=16, depth=2,
                       kernel_size=4, filters=22, dropout=0.3, activation='elu')
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
    x2_sub = tf.keras.layers.Concatenate()([xa, xt, xb])
    x31    = cbam_block_1d(x2_sub)

    # ✅ KEY CHANGE: Plain Concatenate instead of GatedAttentionFusion
    fused  = tf.keras.layers.Concatenate()([x3, x31])

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
    out = Dense(units=4)(out)
    out = Activation('softmax')(out)

    model_no_gated_fusion = Model(inputs=[input_main, input_theta, input_alpha, input_beta], outputs=out)
    return model_no_gated_fusion


def build_main_only(temporal_dimension=500, num_channels=22, num_classes=4):
    input_main = Input(shape=(temporal_dimension, num_channels), name="input_main")

    # --- Main Branch ---
    x_main = Reshape((num_channels, temporal_dimension, 1))(input_main)
    x_main = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_main)
    x      = Reshape((temporal_dimension, num_channels))(x_main)
    x1     = TCN_block(input_layer=input_main, input_dimension=16, depth=2,
                       kernel_size=4, filters=22, dropout=0.3, activation='elu')
    x2     = tf.keras.layers.Concatenate()([x, x1])
    x3     = cbam_block_1d(x2)  # x3 goes directly to output — no fusion

    # --- Output Block ---
    out = Conv1D(32, 5, activation='relu', padding='same')(x3)
    out = MaxPooling1D(2)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.4)(out)
    out = Conv1D(64, 3, activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = GlobalAveragePooling1D()(out)
    out = Dropout(0.3)(out)
    out = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))(out)
    out = Dense(units=4)(out)
    out = Activation('softmax')(out)

    model_main_only = Model(inputs=[input_main], outputs=out)
    return model_main_only


def build_no_alpha(temporal_dimension=500, num_channels=22, num_classes=4):
    input_main  = Input(shape=(temporal_dimension, num_channels), name="input_main")
    input_theta = Input(shape=(temporal_dimension, num_channels), name="input_theta")
    input_beta  = Input(shape=(temporal_dimension, num_channels), name="input_beta")

    # --- Main Branch ---
    x_main = Reshape((num_channels, temporal_dimension, 1))(input_main)
    x_main = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_main)
    x      = Reshape((temporal_dimension, num_channels))(x_main)
    x1     = TCN_block(input_layer=input_main, input_dimension=16, depth=2,
                       kernel_size=4, filters=22, dropout=0.3, activation='elu')
    x2     = tf.keras.layers.Concatenate()([x, x1])
    x3     = cbam_block_1d(x2)

    # --- Theta Branch ---
    x_theta = Reshape((num_channels, temporal_dimension, 1))(input_theta)
    x_theta = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_theta)
    xt      = Reshape((temporal_dimension, num_channels))(x_theta)

    # --- Beta Branch ---
    x_beta = Reshape((num_channels, temporal_dimension, 1))(input_beta)
    x_beta = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_beta)
    xb     = Reshape((temporal_dimension, num_channels))(x_beta)

    # --- Fusion (Theta + Beta only) ---
    x2_sub = tf.keras.layers.Concatenate()([xt, xb])
    x31    = cbam_block_1d(x2_sub)

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
    out = Dense(units=4)(out)
    out = Activation('softmax')(out)

    model_no_alpha = Model(inputs=[input_main, input_theta, input_beta], outputs=out)
    return model_no_alpha


def build_no_beta(temporal_dimension=500, num_channels=22, num_classes=4):
    input_main  = Input(shape=(temporal_dimension, num_channels), name="input_main")
    input_theta = Input(shape=(temporal_dimension, num_channels), name="input_theta")
    input_alpha = Input(shape=(temporal_dimension, num_channels), name="input_alpha")

    # --- Main Branch ---
    x_main = Reshape((num_channels, temporal_dimension, 1))(input_main)
    x_main = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_main)
    x      = Reshape((temporal_dimension, num_channels))(x_main)
    x1     = TCN_block(input_layer=input_main, input_dimension=16, depth=2,
                       kernel_size=4, filters=22, dropout=0.3, activation='elu')
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

    # --- Fusion (Theta + Alpha only) ---
    x2_sub = tf.keras.layers.Concatenate()([xt, xa])
    x31    = cbam_block_1d(x2_sub)

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
    out = Dense(units=4)(out)
    out = Activation('softmax')(out)

    model_no_beta = Model(inputs=[input_main, input_theta, input_alpha], outputs=out)
    return model_no_beta


def build_no_theta(temporal_dimension=500, num_channels=22, num_classes=4):
    input_main  = Input(shape=(temporal_dimension, num_channels), name="input_main")
    input_alpha = Input(shape=(temporal_dimension, num_channels), name="input_alpha")
    input_beta  = Input(shape=(temporal_dimension, num_channels), name="input_beta")

    # --- Main Branch ---
    x_main = Reshape((num_channels, temporal_dimension, 1))(input_main)
    x_main = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_main)
    x      = Reshape((temporal_dimension, num_channels))(x_main)
    x1     = TCN_block(input_layer=input_main, input_dimension=16, depth=2,
                       kernel_size=4, filters=22, dropout=0.3, activation='elu')
    x2     = tf.keras.layers.Concatenate()([x, x1])
    x3     = cbam_block_1d(x2)

    # --- Alpha Branch ---
    x_alpha = Reshape((num_channels, temporal_dimension, 1))(input_alpha)
    x_alpha = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_alpha)
    xa      = Reshape((temporal_dimension, num_channels))(x_alpha)

    # --- Beta Branch ---
    x_beta = Reshape((num_channels, temporal_dimension, 1))(input_beta)
    x_beta = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_beta)
    xb     = Reshape((temporal_dimension, num_channels))(x_beta)

    # --- Fusion (Alpha + Beta only) ---
    x2_sub = tf.keras.layers.Concatenate()([xa, xb])
    x31    = cbam_block_1d(x2_sub)

    fused_dim = int(x3.shape[-1]) + int(x31.shape[-1])
    fused     = GatedAttentionFusion(fused_dim=fused_dim)(x3, x31)

    # --- Output Block (same for all variants) ---
    out = Conv1D(32, 5, activation='relu', padding='same')(fused)
    out = MaxPooling1D(2)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.4)(out)
    out = Conv1D(64, 3, activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = GlobalAveragePooling1D()(out)
    out = Dropout(0.3)(out)
    out = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))(out)
    out = Dense(units=4)(out)
    out = Activation('softmax')(out)

    model_no_theta = Model(inputs=[input_main, input_alpha, input_beta], outputs=out)
    return model_no_theta


def build_subbands_only(temporal_dimension=500, num_channels=22, num_classes=4):
    input_theta = Input(shape=(temporal_dimension, num_channels), name="input_theta")
    input_alpha = Input(shape=(temporal_dimension, num_channels), name="input_alpha")
    input_beta  = Input(shape=(temporal_dimension, num_channels), name="input_beta")

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

    # --- Sub-band Fusion directly to output ---
    x2_sub = tf.keras.layers.Concatenate()([xa, xt, xb])
    x31    = cbam_block_1d(x2_sub)

    # --- Output Block ---
    out = Conv1D(32, 5, activation='relu', padding='same')(x31)
    out = MaxPooling1D(2)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.4)(out)
    out = Conv1D(64, 3, activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = GlobalAveragePooling1D()(out)
    out = Dropout(0.3)(out)
    out = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))(out)
    out = Dense(units=4)(out)
    out = Activation('softmax')(out)

    model_subbands_only = Model(inputs=[input_theta, input_alpha, input_beta], outputs=out)
    return model_subbands_only


def build_full(temporal_dimension=500, num_channels=22, num_classes=4):
    input_main  = Input(shape=(temporal_dimension, num_channels), name="input_main")
    input_theta = Input(shape=(temporal_dimension, num_channels), name="input_theta")
    input_alpha = Input(shape=(temporal_dimension, num_channels), name="input_alpha")
    input_beta  = Input(shape=(temporal_dimension, num_channels), name="input_beta")

    # --- Main Branch ---
    x_main = Reshape((num_channels, temporal_dimension, 1))(input_main)
    x_main = MyAttention(num_ch=num_channels, num_t=temporal_dimension)(x_main)
    x      = Reshape((temporal_dimension, num_channels))(x_main)

    x1     = TCN_block(input_layer=input_main, input_dimension=16, depth=2,
                       kernel_size=4, filters=22, dropout=0.3, activation='elu')

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

    # --- Fusion ---
    x2_sub = tf.keras.layers.Concatenate()([xa, xt, xb])
    x31    = cbam_block_1d(x2_sub)

    # Calculate fused dimensions dynamically
    fused_dim = int(x3.shape[-1]) + int(x31.shape[-1])
    fused     = GatedAttentionFusion5(fused_dim=fused_dim)(x3, x31)

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
    out = Dense(units=4)(out)
    out = Activation('softmax')(out)

    modellstm = Model(inputs=[input_main, input_theta, input_alpha, input_beta], outputs=out)
    return modellstm

BUILDERS = {
    "full": build_full, "additive": build_additive,
    "concatenation": build_concatenation, "main-only": build_main_only,
    "subbands-only": build_subbands_only, "no-alpha": build_no_alpha,
    "no-beta": build_no_beta, "no-theta": build_no_theta,
}
def build_model(dataset="2a", variant="full"):
    if dataset == "2b":
        if variant != "full":
            raise ValueError("The supplied 2b notebook only defines the full model")
        from model_2b import build_full as build_2b
        return build_2b()
    return BUILDERS[variant]()
