# Machine learning intended for doing training, validation, and testing first on Google data set 
# from https://github.com/google-research/google-research/tree/master/goemotions
# then try it on on database text (w/ attention weights). This is a WIP.

import keras
import tensorflow as tf
from tcn import TCN, tcn_full_summary
from keras.layers import Input, Dropout, Softmax, Embedding, \
                         LayerNormalization, Attention, Dense, GlobalAveragePooling1D

from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping


# x_tr, y_tr = [], []
# x_val, y_val = [], []
# num_classes = 28
# num_neurons_de1 = 15
# num_neurons_de2 = 15

# input_dim = 30
# output_dim = num_classes 

class TCN_Mood():

    def __init__(self, input_dim, output_dim) -> None:
        self.input_dim_ = input_dim
        self.outout_dim_ = output_dim

    def construct_model(self, num_neurons_de1, num_neurons_de2, dropout = 0.1, dilations=[1, 2, 4, 8, 16, 32], opt = keras.optimizers.Adam(learning_rate=0.0001)):
        # Variable-length int sequences.
        query_input = Input(shape=(None,), dtype='string')
        value_input = Input(shape=(None,), dtype='int32')

        # Embedding lookup.
        token_embedding = Embedding(input_dim=self.input_dim_, output_dim=self.outout_dim_)
        # Query embeddings of shape [batch_size, Tq, dimension].
        query_embeddings = token_embedding(query_input)
        # Value embeddings of shape [batch_size, Tv, dimension].
        value_embeddings = token_embedding(value_input)

        t1 = TCN(return_sequences=False, 
                    dilations=dilations, 
                    dropout_rate = dropout, 
                    use_skip_connections=True, 
                    padding='causal', 
                    use_layer_norm=True)

        # Query encoding of shape [batch_size, Tq, filters].
        query_seq_encoding = t1(query_embeddings)
        # Value encoding of shape [batch_size, Tv, filters].
        value_seq_encoding = t1(value_embeddings)

        # Query-value attention of shape [batch_size, Tq, filters].
        query_value_attention_seq = tf.keras.layers.Attention()(
            [query_seq_encoding, value_seq_encoding])

        # Reduce over the sequence axis to produce encodings of shape
        # [batch_size, filters].
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
            query_seq_encoding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
            query_value_attention_seq)

        # Concatenate query and document encodings to produce a DNN input layer.
        input_layer = tf.keras.layers.Concatenate()(
            [query_encoding, query_value_attention])

        ln1 = LayerNormalization()(input_layer)
        dr1 = Dropout(dropout)(ln1)

        de1 = Dense(num_neurons_de1, activation='relu')(dr1)# to do: vary number of neurons
        ln2 = LayerNormalization()(de1)
        dr2 = Dropout(dropout)(ln2)
        de2 = Dense(num_neurons_de2, activation='relu')(dr2)
        ln3 = LayerNormalization()(de2)
        dr3 = Dropout(dropout)(ln3)
        o = Softmax(self.output_dim)(dr3)

        self.m = Model(inputs=[query_input], outputs=[o])

        self.m.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy(), metrics=['accuracy'] )


    def train_model(self, x_tr, y_tr, 
                    val_split=0.20, 
                    callback_loss = EarlyStopping(monitor='val_loss', min_delta= 0.01, patience=10, mode='auto', restore_best_weights=True),
                    callback_acc = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=10, mode='auto', restore_best_weights=True)):
        
        self.train_result = self.m.fit(x_tr, y_tr, epochs=250, validation_split=val_split, callbacks = [callback_loss, callback_acc])

    def save_model(self, path):
        self.m.save(path)

    def load_model(self, path):
        self.m.load(path)

#The emotion categories are: admiration, amusement, anger, annoyance, approval, caring,
#  confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise.
 