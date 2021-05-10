import imgaug as ia
from imgaug import augmenters as iaa
from keras_video import VideoFrameGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

class CustomVideoGenerator(VideoFrameGenerator):
    def __init__(self, *args, **kwargs):
        super(CustomVideoGenerator, self).__init__(*args, **kwargs)
        self.__frame_cache = {}
        self.img_aug = kwargs.get('img_aug', False)
        self.skin_normalize = kwargs.get('skin_normalize', True)


    def RGB_Skin_Rule(self, image):
        red, green, blue = cv2.split(image)
        BRG_MAX = np.maximum.reduce([red, green, blue])
        BRG_MIN = np.minimum.reduce([red, green, blue])
        RGB_Rule_1 = np.logical_and.reduce([red > 95, green > 40, blue > 20, \
                                        BRG_MAX - BRG_MIN > 15, abs(red - green) > 15, \
                                        red > green, red > blue])
        RGB_Rule_2 = np.logical_and.reduce([red > 220, green > 210, blue > 170, \
                                            abs(red - green) <= 15, red > blue, green > blue])
        RGB_Rule_All = np.logical_not(np.logical_or(RGB_Rule_1, RGB_Rule_2))
        return RGB_Rule_All

    
    def lines(self, axis):
        line1 = 1.5862 * axis + 20
        line2 = 0.3448 * axis + 76.2069
        line3 = -1.005 * axis + 234.5652
        line4 = -1.15 * axis + 301.75
        line5 = -2.2857 * axis + 432.85
        return line1, line2, line3, line4, line5


    def YCrCb_Skin_Rule(self, image):
        YCrCb_FRAME = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        Y_FRAME, Cr_FRAME, Cb_FRAME = [YCrCb_FRAME[..., YCrCb] for YCrCb in range(3)]
        line1, line2, line3, line4, line5 = self.lines(Cb_FRAME)
        YCrCb_Rule_All = np.logical_and.reduce([line1 - Cr_FRAME >= 0, \
                                            line2 - Cr_FRAME <= 0,
                                            line3 - Cr_FRAME <= 0,
                                            line4 - Cr_FRAME >= 0,
                                            line5 - Cr_FRAME >= 0])
        YCrCb_Rule_All = np.logical_not(YCrCb_Rule_All)
        return YCrCb_Rule_All


    def HSV_Skin_Rule(self, image):
        HSV_FRAME = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        H_FRAME, _, _ = cv2.split(image)
        HSV_Rule = np.logical_and.reduce([H_FRAME < 50, H_FRAME > 150])
        HSV_Rule = np.logical_not(HSV_Rule)
        return HSV_Rule


    def skin_detection(self, image):
        RGB_Rule_All = self.RGB_Skin_Rule(image)
        YCrCb_Rule_All = self.YCrCb_Skin_Rule(image)
        HSV_Rule_All = self.HSV_Skin_Rule(image)
        Rule_All = np.logical_and(RGB_Rule_All, YCrCb_Rule_All)
        Rule_All = np.logical_and(Rule_All, HSV_Rule_All)
        image[Rule_All] = 0
        return image


    def __getitem__(self, index):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        transformation = None

        for i in indexes:
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            video = self.files[i]
            classname = self._get_classname(video)

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.

            if video not in self.__frame_cache:
                frames = self._get_frames(
                    video,
                    nbframe,
                    shape,
                    force_no_headers=not self.use_video_header)
                if frames is None:
                    # avoid failure, nevermind that video...
                    continue

                # add to cache
                if self.use_frame_cache:
                    self.__frame_cache[video] = frames

            else:
                frames = self.__frame_cache[video]

            # apply transformation
            frames = [frame * 255 for frame in frames]
            frames = [frame.astype(np.uint8) for frame in frames]
            if self.skin_normalize == True:
                frames = [self.skin_detection(frame) for frame in frames]
            
            if self.img_aug == True:
                frames = self.do_augmentor(frames)

            frames = [frame * self.rescale for frame in frames]
            # add the sequence in batch
            images.append(frames)
            labels.append(label)
        
        return np.array(images), np.array(labels)


    def do_augmentor(self, images):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5,aug)
        seq = iaa.Sequential(
				[
				# apply the following augmenters to most images
				#iaa.Fliplr(0.5),  # horizontally flip 50% of all images
				iaa.Flipud(0.2),  # vertically flip 20% of all images
				sometimes(iaa.Affine(
					scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
					# scale images to 80-120% of their size, individually per axis
					translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
					# translate by -20 to +20 percent (per axis)
					rotate=(-5, 5),  # rotate by -45 to +45 degrees
					shear=(-5, 5),  # shear by -16 to +16 degrees
					order=[0, 1],
					# use nearest neighbour or bilinear interpolation (fast)
					cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
					mode=ia.ALL
					# use any of scikit-image's warping modes (see 2nd image from the top for examples)
				)),
                ],
                random_order=True)
				# execute 0 to 5 of the following (less important) augmenters per image
				# don't execute all of them, as that would often be way too strong

        return seq.augment_images(images)
    

        def get_validation_generator(self):
            """ Return the validation generator if you've provided split factor """
            return self.__class__(
                nb_frames=self.nbframe,
                nb_channel=self.nb_channel,
                target_shape=self.target_shape,
                classes=self.classes,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                rescale=self.rescale,
                glob_pattern=self.glob_pattern,
                use_headers=self.use_video_header,
                _validation_data=self.validation,
                img_aug=False,
                skin_normalize=True)