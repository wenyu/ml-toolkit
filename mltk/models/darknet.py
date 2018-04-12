import keras as K


def Darknet19(input_shape=(224, 224, 3), name="", pooling=None):
    layer_name = lambda s: name + "_" + s

    def conv_bn(suffix, x, filters, kernel_size, strides=1, alpha=0.1):
        suffix = "_" + suffix
        y = K.layers.ZeroPadding2D((kernel_size >> 1), name=layer_name("pad" + suffix))(x)
        y = K.layers.Conv2D(filters, kernel_size, padding="valid", strides=strides, name=layer_name("conv" + suffix))(y)
        y = K.layers.BatchNormalization(name=layer_name("bn" + suffix))(y)
        y = K.layers.LeakyReLU(alpha, name=layer_name("leaky" + suffix))(y)
        return y

    x = K.layers.Input(shape=input_shape, name=layer_name("input"))
    y = conv_bn("1", x, 32, 3, 2)
    y = K.layers.MaxPool2D(strides=2, name=layer_name("pool_1"))(y)
    y = conv_bn("2", y, 64, 3)
    y = K.layers.MaxPool2D(strides=2, name=layer_name("pool_2"))(y)
    y = conv_bn("3.1", y, 128, 3)
    y = conv_bn("3.2", y, 64, 1)
    y = conv_bn("3.3", y, 128, 3)
    y = K.layers.MaxPool2D(strides=2, name=layer_name("pool_3"))(y)
    y = conv_bn("4.1", y, 512, 3)
    y = conv_bn("4.2", y, 256, 1)
    y = conv_bn("4.3", y, 512, 3)
    y = conv_bn("4.4", y, 256, 1)
    y = conv_bn("4.5", y, 512, 3)
    y = K.layers.MaxPool2D(strides=2, name=layer_name("pool_4"))(y)
    y = conv_bn("5.1", y, 1024, 3)
    y = conv_bn("5.2", y, 512, 1)
    y = conv_bn("5.3", y, 1024, 3)
    y = conv_bn("5.4", y, 512, 1)
    y = conv_bn("5.5", y, 1024, 3)
    y = K.layers.Conv2D(1000, 1, name=layer_name("conv_final"))(y)

    if pooling in ["avg", "average"]:
        y = K.layers.GlobalAvgPool2D(name=layer_name("pool_global_avg"))(y)
    elif pooling in ["max", "maximum"]:
        y = K.layers.GlobalAvgPool2D(name=layer_name("pool_global_max"))(y)

    return K.models.Model(x, y)