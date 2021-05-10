import keras
from keras.layers import Conv2D, BatchNormalization, \
    MaxPool2D, GlobalMaxPool2D
from keras.layers import TimeDistributed, GRU, Dense, Dropout, LSTM, Flatten, Input, GlobalAveragePooling1D, GlobalMaxPooling1D
from config import SIZE, CHANNEL, N_FRAME, BATCH_SIZE, EPOCH
from keras.models import Model, Sequential



def build_convnet(shape=(224, 224, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2D(64, (3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool2D())
    return model


def build_mobilenet(shape=(224, 224, 3)):
    model = keras.applications.MobileNetV2(
        include_top=False,
        input_shape=shape,
        weights='imagenet')
    # Keep 9 layers to train﻿﻿ ~ can be 6, 9, 12
    trainable = 6
    for layer in model.layers[:-trainable]:
        layer.trainable = False
    for layer in model.layers[-trainable:]:
        layer.trainable = True
    output = GlobalMaxPool2D()
    return keras.Sequential([model, output])


def build_resnet50v2(shape=(224, 224, 3)):
    model = keras.applications.ResNet50V2(include_top=False, \
                                        input_shape=shape, \
                                        weights='imagenet')
    #
    #trainable = 11
    #for layer in model.layers[:-trainable]:
    #    layer.trainable = False
    #for layer in model.layers[-trainable:]:
    #    layer.trainable = True
    output = GlobalMaxPool2D()
    return keras.Sequential([model, output])


def get_sequence_model(sequence_model='gru'):
    if sequence_model == 'gru':
        return GRU(64)
    if sequence_model == 'lstm':
        return LSTM(1024, activation='relu', return_sequences=True)


def action_model(shape=(10, 224, 224, 3), nbout = 3, feature_extractor='mobilenetv2', sequence_model='gru'):
    # Create our convnet with (224, 224, 3) input shape
    if feature_extractor == 'mobilenetv2':
        convnet = build_mobilenet(shape[1:])
    elif feature_extractor == 'resnet50v2':
        convnet = build_resnet50v2(shape[1:])
    else:
        convnet = build_convnet(shape[1:])
    # then create our final model
    model_input = Input(shape=shape)
    ###model = keras.Sequential()
    # add the convnet with (10, 224, 224, 3) shape
    ###model.add(TimeDistributed(convnet, input_shape=shape))
    feature_extractor_time_distributed = TimeDistributed(convnet, input_shape=shape)(model_input)
    ###model.add(TimeDistributed(Flatten()))
    feature_extractor_time_distributed_flatten = TimeDistributed(Flatten())(feature_extractor_time_distributed)
    # here, you can also use GRU or LSTM
    #model.add(get_sequence_model(sequence_model=sequence_model))
    sequence_model = get_sequence_model(sequence_model=sequence_model)(feature_extractor_time_distributed_flatten)
    #LSTM Output
    ###model.add(TimeDistributed(Dense(128, activation='relu')))
    time_distributed_dense_512 = TimeDistributed(Dense(512, activation='relu'))(sequence_model)

    time_distributed_dense_256 = TimeDistributed(Dense(256, activation='relu'))(time_distributed_dense_512)
    time_distributed_dense_256_drop_out = TimeDistributed(Dropout(0.5))(time_distributed_dense_256)
    
    time_distributed_dense_128 = TimeDistributed(Dense(128, activation='relu'))(time_distributed_dense_256_drop_out)
    time_distributed_dense_128_drop_out = TimeDistributed(Dropout(0.5))(time_distributed_dense_128)
    ###model.add(TimeDistributed(Dense(64, activation='relu')))
    time_distributed_dense_64 = TimeDistributed(Dense(64, activation='relu'))( time_distributed_dense_128_drop_out)
    # Flatten
    ###model.add(Flatten())
    #merged = Flatten()(time_distributed_dense_64)
    # and finally, we make a decision network
    ###model.add(Dense(64, activation='relu'))
    #dense_64 = Dense(64, activation='relu')(merged)
    ###model.add(Dropout(.5))
    ###model.add(Dense(nbout, activation='sigmoid'))
    #prediction = Dense(nbout, activation='sigmoid')(dense_64)
    predictions = TimeDistributed(Dense(nbout, activation='softmax'))(time_distributed_dense_64)
    prediction = GlobalAveragePooling1D()(predictions)
    #predictions = keras.layers.Average()([prediction])
    print(f'Training model to predict {nbout} classes')

    model = Model(model_input, prediction)
    return model
