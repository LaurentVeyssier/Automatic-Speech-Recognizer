from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPool1D, Dropout)
from keras import regularizers

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer                                                 # input (batchs,seqlength,input_dim=161)
    simp_rnn = GRU(units, activation=activation,                          
        return_sequences=True, implementation=2, name='rnn')(input_data)  # output(batchs,seqlength,units = 200)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='Bn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bn_rnn)               # output(batchs,seqlength,output_dim=29)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)  # output(batchs,seqlength,output_dim=29)  seqlength x vector proba distrib
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))   # input (batchs,seqlength,input_dim=161)
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,                          # input (batchs,seqlength = 381,input_dim=161)
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)                     # output(batchs, 186, filters = 200) kernel=11, stride=2, padding=0
    # Add batch normalization                                       # output(batchs, 186, filters = 200) if padding=same
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, name='rnn')(bn_cnn)                  # output(batchs, 186, units = 200)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bn_rnn) # output(batchs, 186, output_dim = 29)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)               # model output length (time steps) defined here at 186 from Convlayer
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    input_layer = input_data
    for layer_index in range(recur_layers):
        simp_rnn = GRU(units=units, return_sequences=True, activation='relu', 
                        implementation=2, name='GRU_{}'.format(layer_index))(input_layer)
        bn_rnn = BatchNormalization(name='bn_rnn_{}'.format(layer_index))(simp_rnn)
        input_layer = bn_rnn
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer                                    # input (batchs,seqlength,input_dim=161)
    bidir_rnn = Bidirectional(GRU(units, activation='relu',                          
                              return_sequences=True, implementation=2),  
                              merge_mode='concat', name='Bidir-GRU')(input_data) # output (batchs,seqlength, 2 x units= 400 concat mode)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bidir_rnn)     # output (batchs,seqlength,output_dim=29)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deep_CONV_RNN_model(input_dim = 161, filters = 200, kernel_size = 11, conv_stride = 2,
    conv_border_mode= 'valid', units = 200, recur_layers=2, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    bn_cnn = BatchNormalization(name='bn_input_1d')(input_data)
    # 1D-Convolutional layer
    conv_1d = Conv1D(filters, kernel_size,                          # input (batchs,seqlength = 381,input_dim=161)
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(bn_cnn)                     # output(batchs, 186, filters = 200) kernel=11, stride=2, padding=0
    # batch normalization                                      
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)         # output(batchs, 186, filters = 200) if padding=same
    
    # Deep RNN with 2 GRU layers, each with batch normalization
    input_layer = bn_cnn
    for layer_index in range(recur_layers):
        simp_rnn = SimpleRNN(units, activation='relu',
                             dropout=0.1, recurrent_dropout=0.1, go_backwards=True,
                             return_sequences=True, name='rnn_{}'.format(layer_index))(input_layer)
        bn_rnn = BatchNormalization(name='bn_rnn_{}'.format(layer_index))(simp_rnn)
        input_layer = bn_rnn
    
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bn_rnn)
    # Softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)               # model output length (time steps) defined here at 186 from Convlayer
    print(model.summary())
    return model

def deep_CONV_GRU_model(input_dim = 161, filters = 200, kernel_size = 11, conv_stride = 2,
    conv_border_mode= 'valid', units = 200, recur_layers=2, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    bn_cnn = BatchNormalization(name='bn_input_1d')(input_data)
    # 1D-Convolutional layer
    conv_1d = Conv1D(filters, kernel_size,                          # input (batchs,seqlength = 381,input_dim=161)
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(bn_cnn)                     # output(batchs, 186, filters = 200) kernel=11, stride=2, padding=0
    # batch normalization                                      
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)         # output(batchs, 186, filters = 200) if padding=same
    
    # Deep RNN with 2 GRU layers, each with batch normalization
    input_layer = bn_cnn
    for layer_index in range(recur_layers):
        simp_rnn = GRU(units=units, return_sequences=True, activation='relu', dropout=0.1, recurrent_dropout=0.1,
                       kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), recurrent_regularizer=regularizers.l2(1e-4),
                       implementation=2, name='GRU_{}'.format(layer_index))(input_layer)
        bn_rnn = BatchNormalization(name='bn_rnn_{}'.format(layer_index))(simp_rnn)
        input_layer = bn_rnn
    
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bn_rnn)
    # Softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)               # model output length (time steps) defined here at 186 from Convlayer
    print(model.summary())
    return model

def deep_bidirectional_GRU_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer                                    # input (batchs,seqlength,input_dim=161)
    layer_1 = Bidirectional(GRU(units, activation='relu',                          
                              return_sequences=True, implementation=2, dropout=0.1),
                            merge_mode='concat', name='BiDir_1-GRU')(input_data) # output (batchs,seqlength, 2 x units= 400 concat mode)
    layer_2 = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, dropout=0.1),
                            merge_mode='concat', name='BiDir_2-GRU')(layer_1) # output (batchs,seqlength, 2 x units= 400 concat mode)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(layer_2)     # output (batchs,seqlength,output_dim=29)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deep_bidirectional_GRU_BN_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer                                    # input (batchs,seqlength,input_dim=161)
    #bn_cnn = BatchNormalization(name='bn_input_1d')(input_data)
    layer_1 = Bidirectional(GRU(units, activation='relu',                          
                              return_sequences=True, implementation=2, dropout=0.1),
                            merge_mode='concat', name='BiDir_1-GRU')(input_data) # output (batchs,seqlength, 2 x units= 400 concat mode)
    bn_cnn = BatchNormalization(name='bn_BirDir_1_1d')(layer_1)
    layer_2 = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, dropout=0.1),
                            merge_mode='concat', name='BiDir_2-GRU')(bn_cnn) # output (batchs,seqlength, 2 x units= 400 concat mode)
    bn_cnn = BatchNormalization(name='bn_BirDir_2_1d')(layer_2)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bn_cnn)     # output (batchs,seqlength,output_dim=29)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deep_CONV_biGRU_bn_model(input_dim = 161, filters = 200, kernel_size = 11, conv_stride = 2,
    conv_border_mode= 'valid', units = 200, recur_layers=2, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    #bn_cnn = BatchNormalization(name='bn_input_1d')(input_data)
    # 1D-Convolutional layer
    conv_1d = Conv1D(filters, kernel_size,                          # input (batchs,seqlength = 381,input_dim=161)
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)                     # output(batchs, 186, filters = 200) kernel=11, stride=2, padding=0
    # batch normalization                                      
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)         # output(batchs, 186, filters = 200) if padding=same
    
    # Deep RNN with 2 GRU layers, each with batch normalization
    input_layer = bn_cnn
    for layer_index in range(recur_layers):
        layer = Bidirectional(GRU(units, activation='relu',                          
                return_sequences=True, implementation=2, dropout=0.2, recurrent_dropout=0.2),
                merge_mode='concat', name='biGRU_{}'.format(layer_index))(input_layer) # output (batchs,seqlength, 2 x units= 400 concat mode)
        bn_rnn = BatchNormalization(name='bn_rnn_{}'.format(layer_index))(layer)
        input_layer = bn_rnn
    
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bn_rnn)
    # Softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)               # model output length (time steps) defined here at 186 from Convlayer
    print(model.summary())
    return model


def deep_MultiCONV_RNN_model(input_dim = 161, filters = 200, kernel_size = 11, conv_stride = 1,
    conv_border_mode= 'valid', conv_layers=2, units = 200, recur_layers=2, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    #bn_cnn = BatchNormalization(name='bn_input_1d')(input_data)
    # 1D-Convolutional layer
    input_layer = input_data
    for layer_index in range(conv_layers):
        conv_1d = Conv1D(filters, kernel_size,                          # input (batchs,seqlength = 381,input_dim=161)
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='Conv1d_{}'.format(layer_index))(input_layer) # output(batchs, repeat (381-kernel)/stride +1, filters = 200)
        # batch normalization                                      
        #bn_cnn = BatchNormalization(name='bn_conv_{}'.format(layer_index))(conv_1d)  
        input_layer = conv_1d
    bn_cnn = BatchNormalization(name='bn_conv_1d')(input_layer)
    pool_size=2
    pool = MaxPool1D(pool_size=pool_size)(bn_cnn)  # timestep dimension reduction = (timestep - pool_size + 1)/stride. stride = pool_size
    pool_post_dropout=Dropout(0.2)(pool)
    # Deep RNN with 2 rnn layers, each with batch normalization
    input_layer = pool_post_dropout
    for layer_index in range(recur_layers):
        simp_rnn = SimpleRNN(units, activation='relu',
                             dropout=0.1, recurrent_dropout=0.1, go_backwards=True,
                             return_sequences=True, name='rnn_{}'.format(layer_index))(input_layer)
        bn_rnn = BatchNormalization(name='bn_rnn_{}'.format(layer_index))(simp_rnn)
        input_layer = bn_rnn
    
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bn_rnn)
    # Softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: multi_cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, conv_layers, pool_size) # model output length (time steps) defined from Convlayer
    print(model.output_length)
    
    print(model.summary())
    return model

def deep_MultiCONV_GRU_model(input_dim = 161, filters = 200, kernel_size = 11, conv_stride = 1,
    conv_border_mode= 'valid', conv_layers=2, units = 200, recur_layers=2, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    #bn_cnn = BatchNormalization(name='bn_input_1d')(input_data)
    # 1D-Convolutional layer
    input_layer = input_data
    for layer_index in range(conv_layers):
        conv_1d = Conv1D(filters, kernel_size,                          # input (batchs,seqlength = 381,input_dim=161)
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='Conv1d_{}'.format(layer_index))(input_layer) # output(batchs, repeat (381-kernel)/stride +1 , filters = 200)
        # batch normalization                                      
        #bn_cnn = BatchNormalization(name='bn_conv_{}'.format(layer_index))(conv_1d) 
        input_layer = conv_1d
    bn_cnn = BatchNormalization(name='bn_conv_1d')(input_layer)  
    pool_size=2
    pool = MaxPool1D(pool_size=pool_size)(bn_cnn) # timestep dimension reduction = (timestep - pool_size + 1)/stride. stride = pool_size
    pool_post_dropout=Dropout(0.2)(pool)
    # Deep RNN with 2 rnn layers, each with batch normalization
    input_layer = pool_post_dropout
    for layer_index in range(recur_layers):
        simp_rnn = GRU(units, activation='relu',
                             dropout=0.3, go_backwards=True,
                             return_sequences=True, name='GRU_{}'.format(layer_index))(input_layer)
        bn_rnn = BatchNormalization(name='bn_rnn_{}'.format(layer_index))(simp_rnn)
        input_layer = bn_rnn
    
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bn_rnn)
    # Softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: multi_cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, conv_layers, pool_size) # model output length (time steps) defined from Convlayer
    print(model.output_length)
    
    print(model.summary())
    return model


def multi_cnn_output_length(input_length, filter_size, border_mode, stride,
                       conv_layers=1, pool_size=0, dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    
    length = input_length
    
    for index in range(conv_layers):
    
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        if border_mode == 'same':
            output_length = length
        elif border_mode == 'valid':
            output_length = length - dilated_filter_size + 1
        
        length = (output_length + stride - 1) // stride
    
    if pool_size!=0:
        length = (length - pool_size +1) / pool_size
      
    return length



def final_model(input_dim = 161, filters = 200, kernel_size = 11, conv_stride = 2,
    conv_border_mode= 'valid', units = 200, recur_layers=2, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    bn_cnn = BatchNormalization(name='bn_input_1d')(input_data)
    # 1D-Convolutional layer
    conv_1d = Conv1D(filters, kernel_size,                          # input (batchs,seqlength = 381,input_dim=161)
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(bn_cnn)                         # output(batchsize, 186, filters) kernel=11, stride=2, padding=0
    # batch normalization                                      
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)         # output(batchsize, 186, filters) if padding=same
    
    # Deep RNN with 2 rnn layers, each with batch normalization
    input_layer = bn_cnn
    for layer_index in range(recur_layers):
        simp_rnn = SimpleRNN(units, activation='relu',
                             dropout=0.1, recurrent_dropout=0.1, go_backwards=True,
                             return_sequences=True, name='rnn_{}'.format(layer_index))(input_layer)
        bn_rnn = BatchNormalization(name='bn_rnn_{}'.format(layer_index))(simp_rnn)
        input_layer = bn_rnn                                              # output(batchsize, 186, units) if padding=same
    
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Dense')(bn_rnn) # output(batchsize, 186, output_dim) if padding=same
    # Softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)               # model output length (time steps) defined here at 186 from Convlayer
    print(model.summary())
    return model