import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString as ia_LineString
from imgaug.augmentables.lines import LineStringsOnImage


def get_fastdraw_aug():
    aug_num = 3
    color_shift = iaa.SomeOf(aug_num, [
        iaa.GaussianBlur(sigma=(0.5, 1.5)),
        iaa.LinearContrast((0.5, 1.5), per_channel=False),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.Add((-10, 10), per_channel=0.2),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(0, iaa.Multiply((0.9, 1.1)))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(1, iaa.Multiply((0.1, 2)))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(2, iaa.Multiply((0.5, 1.5)))),
    ], random_order=True)
    posion_shift = iaa.Sequential([
        iaa.Crop(percent=([0, 0.1], [0, 0.15], [0, 0.1], [0, 0.15]), keep_size=True),
        iaa.Rotate(rotate=(-10, 10)),
        iaa.TranslateX(px=(-16, 16)),
    ])
    aug = iaa.Sequential([
        iaa.Crop(px=((270, 270), (0, 0), (0, 0), (0, 0)), keep_size=True),
        iaa.Fliplr(p=0.5),
        iaa.Sometimes(p=0.7, then_list=color_shift),
        iaa.Sometimes(p=0.7, then_list=posion_shift)
    ], random_order=True)
    return aug

def get_infer_aug():
    return iaa.Crop(px=((270, 270), (0, 0), (0, 0), (0, 0)), keep_size=True)