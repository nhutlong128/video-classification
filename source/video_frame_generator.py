import os
import glob
import keras
from keras_video import VideoFrameGenerator
import keras_video.utils

def generate_video_frame(size=(112,112), channel=3, n_frame=5, batch_size=8):
    # use sub directories names as classes
    classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
    classes.sort()
    # pattern to get videos and classes
    glob_pattern = 'videos/{classname}/*.avi'
    # for data augmentation
    data_aug = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=.1,
        horizontal_flip=True,
        rotation_range=8,
        width_shift_range=.2,
        height_shift_range=.2)
    # Create video frame generator
    train = VideoFrameGenerator(
        classes=classes, 
        glob_pattern=glob_pattern,
        nb_frames=n_frame,
        split_val=.33, 
        shuffle=True,
        batch_size=batch_size,
        target_shape=size,
        nb_channel=channel,
        transformation=data_aug,
        use_frame_cache=True)
    valid = train.get_validation_generator()
    return train, valid, classes


# Show samples from train generator
#keras_video.utils.show_sample(train)
