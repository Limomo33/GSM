import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
import keras
import pickle
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from foutput import *
from adj import *
from model_eval_new import *
from math import log
from model_performance_2inputs import *
from keras.engine.topology import Layer
from GAT import *
from keras.regularizers import l2
from keras.losses import mse
from graph import *
from seq_distance import seq_get
import time

def calEuclidean(A, t):
    loss=mse(A,t)
    loss = tf.reshape(loss, [-1])
    loss = tf.reduce_sum(loss)
    print("---------------------")
    print(loss.shape)

    return loss

class Position_Embedding(Layer):

    def __init__(self, size=8, mode='concat', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000.,2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(Layer):

    def __init__(self, nb_head=4, size_per_head=16, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

def cross_validation_training_transformer_gat(training_data, test_dict, validation_data, validation_target, global_args):
    ########preprocessing##########
    [blosum_matrix, aa, main_dir, output_path] = global_args
    SYM_NORM = True
    [training_pep, training_mhc, training_target] = [[i[j] for i in training_data] for j in range(3)]
    validation_pep, validation_mhc = [i[0] for i in validation_data], [i[1] for i in validation_data]


    print("traindata shape:", np.array(training_mhc).shape, np.array(training_pep).shape)
    print("val data shape", np.array(validation_mhc).shape, np.array(validation_pep).shape)
    #################################################################
    ####### hyperparameters##########
    filters, fc1_size, fc2_size, fc3_size = 128, 256, 64, 2
    foutput(str(filters) + "\t" + str(fc1_size) + "\t" + str(fc2_size) + "\t" + str(fc3_size)+"mse_loss", output_path)

    support=1
    # A1=np.zeros((300,300))
    # A2=np.ones((300,24))
    # A2[:][:10]=-10e5
    # A2[:][10:20]=-10e4
    # A2[:][20:30] = -10e3
    # A2[:][30:40] = -10e2
    # A2[:][40:50] = -10e1
    # for ii in range(1,10):
    #     j=200+ii*10
    #     A2[:][200:j]=-10e1*ii
    # A3=np.transpose(A2)
    # A4=np.zeros((24,24))
    # a=np.vstack((np.hstack((A1,A2)),np.hstack((A3,A4))))
    # A=np.tile(a,(len(training_mhc),1,1))
    # A1=np.zeros((300,300))
    # A2=np.ones((300,12))
    # A3=np.ones((12,300))
    # A4=np.zeros((12,12))
    # a=np.vstack((np.hstack((A1,A2)),np.hstack((A3,A4))))
    # A=np.tile(a,(len(training_mhc),1,1))
    aa=creat_adj()
    aa=np.expand_dims(aa,axis=0)
    aa=aa.astype(np.float32)
    print('a的维度',aa.shape)
    #dd=np.array(aa.sum(1))
    #print('diag_matrix',dd)
    #d=Tran_D(dd,aa)
    # D=np.expand_dims(d,axis=0)
    #D = d.astype(np.float32)
    #d = np.diag(dd)

    A=np.tile(aa,(len(training_mhc),1,1))

    kernel_size = 3
    models = []
    size_n_1 = np.shape(training_pep[0])[0]
    size_n_2 = np.shape(training_mhc[0])[0]

    j=0
    ####new model##############################
    while len(models) < 5:
        inputs_1 = Input(shape=(size_n_1, 20))
        inputs_2 = Input(shape=(size_n_2, 20))
        inputs_4 = Input(shape=(324, 324))
        inputs_3 = Concatenate(axis=1)([inputs_1, inputs_2])

        #################################################self-attention##############################################
        embeddings_1 = Position_Embedding()(inputs_1)
        embeddings_2 = Position_Embedding()(inputs_2)
        O_seq_1 = Attention(4, 16)([embeddings_1, embeddings_1, embeddings_1])
        O_seq_2 = Attention(4, 16)([embeddings_2, embeddings_2, embeddings_2])
        print(K.int_shape(O_seq_1))
        O_seq_1 = AveragePooling1D()(O_seq_1)
        O_seq_2 = AveragePooling1D()(O_seq_2)
        print(K.int_shape(O_seq_1))
        O_seq_1 = Dropout(0.5)(O_seq_1)
        O_seq_2 = Dropout(0.5)(O_seq_2)
        print(K.int_shape(O_seq_1))
        #################################################gat##############################################
        #Test=Concatenate()([embeddings_2,embeddings_1])
        (graph_attention_1, A_1) = GraphAttention2(64,2,dropout_rate=0.5,)(inputs_3,input_4)
        A_2=tf.Variable(tf.constant(1.0,shape=[]),name="A_2")
        #graph_attention=Concatenate()([graph_attention_1,inputs_3])
        (graph_attention_2, A_2) = GraphAttention2(32,2, dropout_rate=0.5,)([graph_attention_1,A_1])
        #graph_attention_2=Concatenate()([graph_attention_2,graph_attention_1])
        # graph_attention_2=Concatenate()([graph_attention_2,graph_attention_1])
        Y=Flatten()(graph_attention_2)
        #H = GraphConvolution(8, support, activation='relu', kernel_regularizer=l2(5e-4))([graph_attention_2, A_2])
        #H = Dropout(0.5)(H)
        #Y = Flatten()(H)
        #Y = Dense(fc1_size, activation="relu")(H)
        ################################################################################################################
        # Initial feature extraction using a convolutional layer
        pep_conv = Conv1D(filters, kernel_size, padding='same', activation='relu', strides=1)(O_seq_1)  #
        # pep_maxpool = MaxPooling1D()(pep_conv)
        mhc_conv_1 = Conv1D(filters, kernel_size, padding='same', activation='relu', strides=1)(O_seq_2)  #

        # The convolutional module
        mhc_conv_2 = Conv1D(filters, kernel_size, padding='same', activation='relu', strides=1)(mhc_conv_1)
        flat_pep_0 = Flatten()(pep_conv)
        flat_pep_1 = Flatten()(pep_conv)
        flat_pep_2 = Flatten()(pep_conv)
        flat_mhc_0 = Flatten()(O_seq_2)  #
        flat_mhc_1 = Flatten()(mhc_conv_1)
        flat_mhc_2 = Flatten()(mhc_conv_2)
        cat_0 = Concatenate()([flat_pep_0, flat_mhc_0])
        cat_1 = Concatenate()([flat_pep_1, flat_mhc_1])
        cat_2 = Concatenate()([flat_pep_2, flat_mhc_2])
        fc1_0 = Dense(fc1_size, activation="relu")(cat_0)
        fc1_1 = Dense(fc1_size, activation="relu")(cat_1)
        fc1_2 = Dense(fc1_size, activation="relu")(cat_2)
        merge_1 = Concatenate()([fc1_0, fc1_1, fc1_2, Y])
        fc2 = Dense(fc2_size, activation="relu")(merge_1)  # merge_1
        fc3 = Dense(1, activation="relu")(fc2)
        # input_4 = Input(shape=(1,))
        # centers = Embedding(20, fc2_size)(input_4)  # Embedding层用来存放中心损失
        # l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([fc2, centers])

        # model = Model(inputs=[inputs_1,inputs_2],outputs=[fc3,A_2])
        model = Model(inputs=[inputs_1, inputs_2, inputs_4], outputs=[fc3])
        model.summary()
        #optimizer=adam(learning_rate=0.0001)
        # model.compile(loss=['mse'],optimizer=keras.optimizers.Adam(lr=0.0002), metrics=['mse'])
        model.compile(loss=['mse', calEuclidean], loss_weights=[0.8, 0.2], optimizer=keras.optimizers.Adam(lr=0.0001),
                      metrics=['mse'])
        poor_init = False

        for n_epoch in range(80):
            start_time=time.time()
            # Start training
            model.fit([np.array(training_pep), np.array(training_mhc),np.array(A)],
                      [np.array(training_target)],
                      batch_size=32,epochs=1)
            end_time=time.time()
            print(str(n_epoch)+':'+str(end_time-start_time))
           ##########################################################################################################################
            pcc, roc_auc, max_acc = model_eval(model,
                                               [np.array(validation_pep), np.array(validation_mhc),np.tile(aa,(len(validation_target),1,1))],
                                               [np.array(validation_target)])
            foutput(str(n_epoch) + "\t" + str(pcc) + "\t" + str(roc_auc) + "\t" + str(max_acc), output_path)
            # If the pcc after the first epoch is very low, the network is unlikely to be trained well, so start again
            if n_epoch == 0 and not ((pcc< 1) and (pcc>0.1)):
                poor_init = True
                break
        if poor_init:
            continue
        if max_acc > 0.8 :
            foutput("Model Adopted", output_path)
            j=j+1
            model_json = model.to_json()
            with open(main_dir + "models/model_724" +str(j) + ".json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights(main_dir + "models/model_724" + str(j) + ".h5")
            model.save(main_dir+"models/model_724 + str(j)" + ".h5")
            models.append(model)

    performance_dict = model_performance(models, test_dict, global_args)

    for i in range(len(models)):
        model = models[i]
        model_json = model.to_json()
        with open(main_dir+ "models/model_724"+str(i)+".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(main_dir+ "models/model_724"+str(i)+".h5")

    return performance_dict
