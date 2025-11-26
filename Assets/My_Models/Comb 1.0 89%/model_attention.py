import tensorflow as tf

def backbone():
    '''
    RETURNS THE BACKBONE FEATURE ENCODER NETWORK
    XCEPTION USED IN THIS CASE
    '''
    mod  = tf.keras.applications.Xception(weights='imagenet')
    mod = tf.keras.Model(mod.input, mod.layers[-13].output)
    return mod
    
class ModifiedBranch(tf.keras.layers.Layer):
    '''
    COMPUTES THE MODIFIED BRANCH TO BE USED IN ATTENTION TECHNIQUE
    '''
    def __init__(self, a_vec_size, **kwargs):
        super(ModifiedBranch, self).__init__(**kwargs)
        self.a_vec_size = a_vec_size

    def build(self, input_shape):
        self.dense_layer = tf.keras.layers.Dense(self.a_vec_size, activation='tanh')

    def call(self, input):
        af = tf.keras.backend.mean(input, axis=2) 
        hs = self.dense_layer(af)
        return hs

class MainBranch(tf.keras.layers.Layer):
    def __init__(self, a_vec_size, dim, **kwargs):
        super(MainBranch, self).__init__(**kwargs)
        self.a_vec_size = a_vec_size
        self.dim = dim

    def build(self, input_shape):
        self.reshape1 = tf.keras.layers.Reshape((-1, self.a_vec_size))
        self.relu = tf.keras.activations.relu
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.reshape2 = tf.keras.layers.Reshape((self.dim**2, self.a_vec_size))

    def call(self, input):
        e = tf.transpose(input, perm=[0, 2, 1])
        e = self.reshape1(e)
        e = self.relu(e)
        e = self.dropout(e)
        e = self.reshape2(e)
        e = tf.transpose(e, perm=[0, 2, 1])
        return e

class Attention(tf.keras.layers.Layer):
    '''
    IMPLEMENTATION OF THE ATTENTION TECHNIQUE ON TWO BRANCHES
    '''
    def __init__(self, dim, a_vec_size, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dim = dim
        self.a_vec_size = a_vec_size
    
    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.dim**2)
        self.reshape1 = tf.keras.layers.Reshape((1, self.dim**2))
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.relu = tf.keras.activations.relu
        self.reshape2 = tf.keras.layers.Reshape((-1, self.a_vec_size))
        self.dense2 = tf.keras.layers.Dense(1, use_bias=False)
        self.reshape3 = tf.keras.layers.Reshape((-1, self.dim**2))

    def call(self, input):
        eh = self.dense1(input[0])
        eh = self.reshape1(eh)
        eh = self.add([input[1], eh])
        eh = self.relu(eh)
        eh = self.dropout(eh)
        eh = tf.transpose(eh, perm=[0, 2, 1])
        eh = self.reshape2(eh)
        eh = self.dense2(eh)
        eh = self.reshape3(eh)
        eh = self.relu(eh)
        return eh

def model(a_vec_size, dim):
    '''
    THIS FUNCTION CALLS THE ENTIRE MODEL
    '''
    back = backbone()
    backbone_feature = back.output  
    out = tf.keras.layers.Conv2D(filters = a_vec_size, kernel_size = (1,1), strides=(1,1), padding = 'valid', use_bias=True)(backbone_feature)
    out = tf.keras.layers.BatchNormalization(axis=-1)(out)
    out = tf.keras.activations.relu(out)
    out = tf.keras.layers.Dropout(0.8)(out)
    out = tf.keras.layers.Reshape((a_vec_size, dim**2))(out)
    
    modified = ModifiedBranch(a_vec_size)(out)
    main = MainBranch(a_vec_size, dim)(out)
    
    # --- CRITICAL MODIFICATION from Plan.txt: Added name='attention_output' ---
    att = Attention(dim, a_vec_size, name="attention_output")([modified, main])
    
    fin = tf.keras.layers.Dense(2, activation='softmax')(att)
    fin = tf.keras.layers.Flatten()(fin)
    mod = tf.keras.Model(inputs=back.input, outputs=fin)
    return mod
