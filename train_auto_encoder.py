import keras as K
from mltk import *
import numpy as np
import random
from PIL import Image, ImageFile
from mltk.models.darknet import Darknet19
import sys

if len(sys.argv) != 3:
    print "Usage:", sys.argv[0], "<PREFIX>", "<DATA_ROOT_PATH>"
    sys.exit()

prefix = sys.argv[1]
paths = path_walker.get_files_within_directory(sys.argv[2], path_walker.TYPE_IMAGE)

def preprocess(x):
    return x / 255.
batch_gen = image_utils.auto_encoder_image_generator(paths,
                                                     size=(256, 256),
                                                     shuffle_paths=True,
                                                     jobs=32,
                                                     preprocessor=preprocess, batch_size=128)


def my_loss(y_true, y_pred):
    diff = K.backend.abs(y_true - y_pred)
    return K.backend.mean(diff + 2 * K.backend.square(diff))


K.losses.my_loss = my_loss
model = Darknet19((256, 256, 3), prefix)
model.summary()


stages = map(lambda (l, k): (prefix + "_" + l, k), [
    ("act_2", 2),
    ("act_3.3", 4),
    ("act_4.5", 8),
    ("act_5.5", 16),
    ("act_6.5", 32),
])
last_stage = False


for stage, kernel_size in stages:
    l = model.get_layer(stage)
    y = K.layers.Deconv2D(3, kernel_size=kernel_size, strides=kernel_size, activation="sigmoid")(l.output)
    ae_model = K.models.Model(model.input, y)
    ae_model.summary()

    if last_stage:
        trainable = False
        for l in ae_model.layers:
            l.trainable = trainable
            if l.name == last_stage:
                trainable = True

        ae_model.compile(optimizer="adam", loss=my_loss)
        ae_model.fit_generator(batch_gen, steps_per_epoch=128, epochs=16)

    for l in ae_model.layers:
        l.trainable = True

    ae_model.compile(optimizer="adam", loss=my_loss)
    ae_model.fit_generator(batch_gen, steps_per_epoch=128, epochs=64)
    last_stage = stage

    for l in ae_model.layers:
        l.trainable = False
    ae_model.compile(optimizer="adam", loss="mse")
    ae_model.save(stage + "_deconv.h5")
    model.save(prefix + ".h5")
