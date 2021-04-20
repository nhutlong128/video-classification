import os
import glob
import keras
from source.video_frame_generator import generate_video_frame
from source.model import action_model
from config import SIZE, CHANNEL, N_FRAME, BATCH_SIZE, EPOCH

# Get train and valid sample
train, valid, classes = generate_video_frame(size=SIZE, channel=CHANNEL, n_frame=N_FRAME, batch_size=BATCH_SIZE)


# Compile Model
INSHAPE = (N_FRAME,) + SIZE + (CHANNEL,) # (N_FRAME, 224, 224, 3)
model = action_model(INSHAPE, len(classes), 'resnet50v2')
optimizer = keras.optimizers.Adam(0.001)
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

# Train Model
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





