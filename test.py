from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dense, GlobalAveragePooling2D, \
    BatchNormalization, Dropout, Reshape, Multiply, Conv2DTranspose, Concatenate, Conv1D, UpSampling2D, \
    AvgPool2D, Lambda, SeparableConv2D, Add, InputSpec, Layer
from keras.models import Model
from keras.optimizers import Adam
from acnet_repvgg_dbb_block_utils import DBB
import numpy as np
import tensorflow as tf


# tf_config = tf.compat.v1.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# tf.compat.v1.Session(config=tf_config)


class MyModel(object):
    def __init__(self, input_shape=(28, 28, 1), output_class=10, stage='train'):
        self.input_shape = input_shape

        self.base_filters = 32
        self.output_class = output_class
        self.stage = stage
        self.dbb = DBB(stage)

    def build_model(self):
        input = Input(shape=self.input_shape, name='input')
        x = input
        for i in range(1, 3):

            # ACNet Block
            name = 'dbb_asym_%d' % i
            x = self.dbb.dbb_asym(x, i * self.base_filters, kernel_size=3, name=name, use_bn=True, model='Add')

            # Diverse Branch Block: Building a Convolution as an Inception-like Unit
            name = 'dbb_dbb_%d' % i
            x = self.dbb.dbb_dbb(x, i * self.base_filters, kernel_size=3, name=name, use_bias=True, use_bn=True)

            # RepVGG: Making VGG-style ConvNets Great Again block
            name = 'rep_vgg_%d' % i
            x = self.dbb.rep_vgg(x, i * self.base_filters, kernel_size=3, name=name, dilation_rate=1,
                                 use_bias=False, use_bn=True, model='Add', padding='same')

            x = MaxPooling2D(name='ddb_kxk_kxk_maxpool_%d' % i)(x)

        x = GlobalAveragePooling2D(name='GlobalAveragePooling2D')(x)
        output = Dense(self.output_class, activation='softmax', name='output')(x)

        return Model(input, output)


if __name__ == '__main__':
    from keras.datasets import mnist
    from keras.utils.np_utils import to_categorical

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = MyModel(input_shape=(28, 28, 1), output_class=10).build_model()
    class_infer = MyModel(input_shape=(28, 28, 1), output_class=10, stage='test')
    model_infer = class_infer.build_model()

    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=640, verbose=1)
    model.save_weights('model.hdf5')

    model.load_weights('model.hdf5')

    model_infer.load_weights('model.hdf5', by_name=True, skip_mismatch=True)

    class_infer.dbb.fusion(model, model_infer)

    pred_y = model.predict(x_test)
    print(pred_y[2:5])
    print('*'*20)
    pred_y = model_infer.predict(x_test)
    print(pred_y[2:5])
