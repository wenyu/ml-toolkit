from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from annotation_utils import object_coordinate_to_bounding_boxes, bounding_box_to_object_coordinate
import generators
import random

__often = lambda aug: iaa.Sometimes(0.7, aug)
__sometimes = lambda aug: iaa.Sometimes(0.5, aug)
__occasionally = lambda aug: iaa.Sometimes(0.3, aug)
__seldom = lambda aug: iaa.Sometimes(0.1, aug)
__rarely = lambda aug: iaa.Sometimes(0.05, aug)

__augseq_shape = [
    iaa.Fliplr(0.5), # horizontally flip 50% of all images
    iaa.Flipud(0.1), # vertically flip 10% of all images
    # crop images by -5% to 10% of their height/width
    __sometimes(iaa.CropAndPad(
        percent=(-0.05, 0.1),
        pad_mode=ia.ALL,
        pad_cval=(0, 255)
    )),
    __sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    )),
    __occasionally(iaa.OneOf([
        iaa.Affine(rotate=(-30, 30), mode=ia.ALL),
        iaa.Affine(shear=(-15, 15), mode=ia.ALL),
    ])),
    __seldom(iaa.OneOf([  # Occasionally rotate image 90 degress clockwise or counter-clockwise
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=-90),
    ])),
    __seldom(iaa.PerspectiveTransform(scale=(0.05, 0.125))),
    __rarely(iaa.PiecewiseAffine(scale=(0.02, 0.05), nb_rows=(4, 6), nb_cols=(4, 6)))
]
__augseq_color = [
    __often(iaa.OneOf([
        iaa.Add((-30, 30), per_channel=0.3),  # change brightness of images (by -30 to 30 of original value)
        iaa.AddToHueAndSaturation((-40, 40)),  # change hue and saturation
    ])),
    __occasionally(iaa.Multiply((0.5, 1.6), per_channel=0.3)),  # change the brightness with multiplication
    iaa.Invert(0.005, per_channel=True),  # Invert Images
    __seldom(iaa.Grayscale(1.0)),
    __occasionally(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.3)),
    __rarely(iaa.Sharpen(alpha=(0.5, 1), lightness=(0.75, 1.5))),
    __rarely(iaa.Emboss(alpha=1, strength=(0, 0.5)))
]
__augseq_noise = [
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.25, 1.75)),
        iaa.AverageBlur(k=(3, 7)),
        iaa.MedianBlur(k=(3, 7)),
        iaa.ElasticTransformation(alpha=(0.1, 2.0), sigma=(0.1, 0.3)),
        iaa.Superpixels(p_replace=0.5, n_segments=(125, 200)),
    ]),
    iaa.EdgeDetect(alpha=(0, 0.35)),
    iaa.AdditiveGaussianNoise(scale=(0.0, 255 * 0.1), per_channel=0.5),
    iaa.OneOf([
        iaa.CoarseDropout((0, 0.125), size_percent=(0.05, 3), per_channel=0.5),
        iaa.Dropout((0, 0.125), per_channel=0.5)
    ])
]

AUGMENT_SHAPE_ONLY = iaa.Sequential(__augseq_shape, random_order=True)
AUGMENT_NORMAL = iaa.Sequential(__augseq_color + __augseq_shape, random_order=True)
AUGMENT_EXTRA = iaa.Sequential(
    __augseq_color + __augseq_shape + [__often(iaa.SomeOf((0, 2), __augseq_noise, random_order=True))],
    random_order=True)


def load_image(path):
    return Image.open(path).convert("RGB")  # type: PIL.Image


def augment_image(img, annotations=None, augmenter=AUGMENT_NORMAL):
    """
    Augment image and annotations

    :type img: numpy.ndarray | PIL.Image.Image
    :type annotations: list[dict] | None
    :type augmenter: imgaug.augmenters.Sequential
    :return: (numpy.ndarray | PIL.Image.Image, list[dict]) | numpy.ndarray | PIL.Image.Image
    """
    return_PIL_Image = False
    if type(img) is Image.Image:
        img = np.array(img)
        return_PIL_Image = True

    det = augmenter.to_deterministic()
    ret_img = det.augment_image(img)
    if return_PIL_Image:
        ret_img = Image.fromarray(ret_img)

    if annotations is not None:
        shape = img.shape[:2]
        bboxes = ia.BoundingBoxesOnImage(map(object_coordinate_to_bounding_boxes, annotations), shape=shape)
        bboxes = det.augment_bounding_boxes([bboxes])[0]
        for annotation, bbox in zip(annotations, bboxes.bounding_boxes):
            annotation["coordinates"] = bounding_box_to_object_coordinate(bbox)

        return ret_img, annotations

    return ret_img


def resize_with_short_side_restriction(img, side=299):
    """
    Resize an image, keep the aspect ratio, but with short side restriction.

    :type img: PIL.Image.Image
    :type side: int
    :return: PIL.Image.Image
    """
    if img.width <= img.height:
        W, H = side, side * img.height / img.width
    else:
        W, H = side * img.width / img.height, side
    return img.resize((W, H))


def resize_with_long_side_restriction(img, side=299):
    """
    Resize an image, keep the aspect ratio, but with long side restriction.

    :type img: PIL.Image.Image
    :type side: int
    :return: PIL.Image.Image
    """
    if img.width <= img.height:
        W, H = side * img.width / img.height, side
    else:
        W, H = side, side * img.height / img.width
    return img.resize((W, H))


def __PTIG_process(args):
    path, size, augment, preprocessor = args
    img = load_image(path)
    if size is not None:
        img = img.resize(size)
    img = np.array(img, float)
    if augment == 1:
        img = augment_image(img, augmenter=AUGMENT_SHAPE_ONLY)
    elif augment in [2, True]:
        img = augment_image(img, augmenter=AUGMENT_NORMAL)
    elif augment == 3:
        img = augment_image(img, augmenter=AUGMENT_EXTRA)
    if preprocessor is not None:
        img = preprocessor(img)
    return path, img


def path_to_image_generator(paths, size=None, jobs=1, preprocessor=None, augment=False, shuffle_paths=False):
    """
    Yields (str, numpy.ndarray) as (path, image)
    """
    if shuffle_paths:
        def shuffle_list(l):
            t = list(l)
            random.shuffle(t)
            return t
        path_gen = generators.fault_tolerant_endless_generator(lambda: shuffle_list(paths))
    else:
        path_gen = generators.fault_tolerant_endless_generator(lambda: paths)

    args = ((path, size, augment, preprocessor) for path in path_gen)

    return generators.fault_tolerant_endless_generator(
        lambda: generators.parallel_map_generator(__PTIG_process, args, jobs, jobs << 2))


def auto_encoder_image_generator(paths, batch_size=64, *args, **kwargs):
    img_pair_gen = ((img, img) for path, img in path_to_image_generator(paths, *args, **kwargs))
    return generators.batch_x_y_generator(img_pair_gen, batch_size)
