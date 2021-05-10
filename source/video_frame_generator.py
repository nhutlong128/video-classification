import os
import glob
import keras
from keras_video import VideoFrameGenerator
from data.CustomVideoGenerator import CustomVideoGenerator
import keras_video.utils
from matplotlib import pyplot as plt
import numpy as np
import splitfolders

main_dir = 'videos/'
output_dir = 'train_valid_dataset/'
splitfolders.ratio(main_dir, output=output_dir, seed=42, ratio=(.8, .2))


def generate_video_frame(size=(224,224), channel=3, n_frame=5, batch_size=8):
    # use sub directories names as classes
    classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
    classes.sort()
    # pattern to get videos and classes
    glob_pattern_train = 'train_valid_dataset/train/{classname}/*.mp4'
    glob_pattern_val = 'train_valid_dataset/val/{classname}/*.mp4'
    # Create video frame generator
    train = CustomVideoGenerator(
        classes=classes, 
        glob_pattern= glob_pattern_train,
        nb_frames=n_frame, 
        shuffle=True,
        batch_size=batch_size,
        target_shape=size,
        nb_channel=channel,
        transformation=None,
        use_frame_cache=True,
        img_aug=True,
        skin_normalize=True)
    valid = CustomVideoGenerator(
        classes=classes, 
        glob_pattern=glob_pattern_val,
        nb_frames=n_frame, 
        shuffle=True,
        batch_size=batch_size,
        target_shape=size,
        nb_channel=channel,
        transformation=None,
        use_frame_cache=True,
        img_aug=False,
        skin_normalize=True)
    return train, valid, classes


# Show samples from train generator
#keras_video.utils.show_sample(train)
if __name__ == "__main__":
    train, valid, _ = generate_video_frame()
    x,y = valid.next()
    for batch in x:
        for img in batch:
            print(img)
            plt.imshow(img)
            plt.show()
