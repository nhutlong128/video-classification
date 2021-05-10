import os
import glob
import keras
from source.video_frame_generator import generate_video_frame
from source.model import action_model
from config import SIZE, CHANNEL, N_FRAME, BATCH_SIZE, EPOCH
import argparse


def train_model(feature_extractor='convnet', sequence_model='gru'):
    # Get train and valid sample
    train, valid, classes = generate_video_frame(size=SIZE, channel=CHANNEL, n_frame=N_FRAME, batch_size=BATCH_SIZE)
    # Compile Model
    INSHAPE = (N_FRAME,) + SIZE + (CHANNEL,) # (N_FRAME, 224, 224, 3)
    model = action_model(INSHAPE, len(classes), feature_extractor=feature_extractor, sequence_model=sequence_model)
    optimizer = keras.optimizers.Adam(0.0001)
    model.compile(
        optimizer,
        'categorical_crossentropy',
        metrics=['acc']
    )
    # Define Callback ~ Trigger each iteration
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(verbose=1),
        keras.callbacks.ModelCheckpoint(
            'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            verbose=1),
    ]
    model.fit(
        train,
        validation_data=valid,
        verbose=1,
        epochs=EPOCH,
        callbacks=callbacks
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--feature-extractor", required=False, type=str, default='resnet50v2',
        help="choose a feature-extractor to extract features from images")
    ap.add_argument("-a", "--sequence-model", required=False, type=str, default='lstm',
        help="choose a model to classify action")
    args = vars(ap.parse_args())
    # Get arguments
    feature_extractor = args['feature_extractor']
    sequence_model = args['sequence_model']
    # Train model
    train_model(feature_extractor=feature_extractor, sequence_model=sequence_model)





