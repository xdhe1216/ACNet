from keras.layers import Conv2D, BatchNormalization, Add, Concatenate, AveragePooling2D
import numpy as np
import keras.backend as K
import tensorflow as tf


def diff_model(model, weight, bias):
    if model == 'Add':
        weight = np.array(weight).sum(axis=0)
        bias = np.array(bias).sum(axis=0)
    elif model == 'Concate':
        weight = np.concatenate(np.array(weight), axis=-1)
        bias = np.concatenate(np.array(bias), axis=-1)

    return weight, bias


def fusion_1x1_kxk(AC_names, trained_model, infer_model):
    """

               |
               |
             1x1
               |
               |
              kxk
               |
               |

    Diverse Branch Block
    """
    for layer_name, use_bias, use_bn, model, epoch in AC_names:
        conv_1x1_weights = trained_model.get_layer(layer_name+'_conv_1x1').get_weights()[0]
        conv_kxk_weights = trained_model.get_layer(layer_name+'_conv_kxk').get_weights()[0]
        if use_bias:
            conv_kxk_bias = trained_model.get_layer(layer_name+'_conv_kxk').get_weights()[1]
            conv_1x1_bias = trained_model.get_layer(layer_name+'_conv_1x1').get_weights()[1]
        else:
            conv_kxk_bias = np.zeros((conv_kxk_weights.shape[-1],))
            conv_1x1_bias = np.zeros((conv_1x1_weights.shape[-1],))

        if use_bn:
            gammas_1x1, betas_1x1, means_1x1, var_1x1 = trained_model.get_layer(layer_name + '_bn_1').get_weights()
            gammas_kxk, betas_kxk, means_kxk, var_kxk = trained_model.get_layer(layer_name + '_bn_2').get_weights()
        else:
            gammas_1x1, betas_1x1, means_1x1, var_1x1 = [np.ones((conv_1x1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_weights.shape[-1],)),
                                                         np.ones((conv_1x1_weights.shape[-1],))]

            gammas_kxk, betas_kxk, means_kxk, var_kxk = [np.ones((conv_kxk_weights.shape[-1],)),
                                                         np.zeros((conv_kxk_weights.shape[-1],)),
                                                         np.zeros((conv_kxk_weights.shape[-1],)),
                                                         np.ones((conv_kxk_weights.shape[-1],))]

        w_1x1 = ((gammas_1x1 / np.sqrt(np.add(var_1x1, 1e-10))) * conv_1x1_weights).transpose([0, 1, 3, 2])
        w_kxk = ((gammas_kxk / np.sqrt(np.add(var_kxk, 1e-10))) * conv_kxk_weights).transpose([0, 1, 3, 2])

        b_1x1 = (((conv_1x1_bias - means_1x1) * gammas_1x1) / np.sqrt(np.add(var_1x1, 1e-10))) + betas_1x1
        b_kxk = (((conv_kxk_bias - means_kxk) * gammas_kxk) / np.sqrt(np.add(var_kxk, 1e-10))) + betas_kxk

        with tf.Session() as sess:
            conv_1x1 = tf.convert_to_tensor(w_1x1)
            conv_kxk = tf.convert_to_tensor(w_kxk)
            numpy_w = K.conv2d(conv_kxk, conv_1x1, padding='same').eval()

        weight = numpy_w.transpose([0, 1, 3, 2])
        bias = np.sum(w_kxk*b_1x1, axis=(0, 1, 3)) + b_kxk

        infer_model.get_layer(layer_name).set_weights([weight, bias])


def fusion_diff_kernel_size(AC_names, trained_model, infer_model):
    """
                |
                |
    -----------------------------
    |           |               |
    |           |               |
    1x1         3x3             5x5
    |           |               |
    |           |               |
    BN          ...             BN
    |           |               |
    |           |               |
    -----------combine--------------
                |
                |
    Diverse Branch Block
    """
    for layer_name, use_bias, use_bn, model, epoch in AC_names:

        weight = []
        bias = []
        for k in range(epoch):

            conv_kxk_weights = trained_model.get_layer(layer_name + '_conv_%d' % k).get_weights()[0]
            if use_bias:
                conv_kxk_bias = trained_model.get_layer(layer_name + '_conv_%d' % k).get_weights()[1]
            else:
                conv_kxk_bias = np.zeros((conv_kxk_weights.shape[-1],))

            if use_bn:
                bn = trained_model.get_layer(layer_name + '_bn_%d' % k).get_weights()
                gammas = bn[0]
                betas = bn[1]
                means = bn[2]
                var = bn[3]
            else:
                means, var, gammas, betas = [np.zeros((conv_kxk_weights.shape[-1],)),
                                             np.ones((conv_kxk_weights.shape[-1],)),
                                             np.ones((conv_kxk_weights.shape[-1],)),
                                             np.zeros((conv_kxk_weights.shape[-1],))]

            weight.append((gammas / np.sqrt(np.add(var, 1e-10))) * conv_kxk_weights)
            bias.append((((conv_kxk_bias - means) * gammas) / np.sqrt(np.add(var, 1e-10))) + betas)
        ws = np.zeros_like(weight[-1])
        bs = np.zeros_like(bias[-1])
        kernel_size = weight[-1].shape[0]
        for k in range(epoch):
            ws[kernel_size//2-k:kernel_size//2+k + 1, kernel_size//2-k:kernel_size//2+k + 1, :, :] += weight[k]
            bs += bias[k]

        infer_model.get_layer(layer_name).set_weights([ws, bs])


def fusion_same_kernel_size(AC_names, trained_model, infer_model):
    """
                |
                |
    -----------------------------
    |           |               |
    |           |               |
    kxk         ...             kxk
    |           |               |
    |           |               |
    BN          ...             BN
    |           |               |
    |           |               |
    -----------combine-----------
                |
                |
    Diverse Branch Block
    """
    for layer_name, use_bias, use_bn, model, epoch in AC_names:

        weight = []
        bias = []
        for k in range(epoch):

            conv_kxk_weights = trained_model.get_layer(layer_name + '_conv_%d' % k).get_weights()[0]
            if use_bias:
                conv_kxk_bias = trained_model.get_layer(layer_name + '_conv_%d' % k).get_weights()[1]
            else:
                conv_kxk_bias = np.zeros((conv_kxk_weights.shape[-1],))

            if use_bn:
                bn = trained_model.get_layer(layer_name + '_bn_%d' % k).get_weights()
                gammas = bn[0]
                betas = bn[1]
                means = bn[2]
                var = bn[3]
            else:
                means, var, gammas, betas = [np.zeros((conv_kxk_weights.shape[-1],)),
                                             np.ones((conv_kxk_weights.shape[-1],)),
                                             np.ones((conv_kxk_weights.shape[-1],)),
                                             np.zeros((conv_kxk_weights.shape[-1],))]

            weight.append((gammas / np.sqrt(np.add(var, 1e-10))) * conv_kxk_weights)
            bias.append((((conv_kxk_bias - means) * gammas) / np.sqrt(np.add(var, 1e-10))) + betas)

        infer_model.get_layer(layer_name).set_weights(diff_model(model, weight, bias))


def fusion_asym(AC_names, trained_model, infer_model):
    """
                    |
                    |
        -----------------------------
        |           |               |
        |           |               |
        1xk         kx1             kxk
        |           |               |
        |           |               |
        BN          BN             BN
        |           |               |
        |           |               |
        -----------combine-----------
                    |
                    |
        Diverse Branch Block
        """
    for layer_name, use_bias, use_bn, model, epoch in AC_names:

        weights = []
        bias = []

        weights.append(trained_model.get_layer(layer_name + '_conv_kxk').get_weights()[0])
        weights.append(trained_model.get_layer(layer_name + '_conv_kx1').get_weights()[0])
        weights.append(trained_model.get_layer(layer_name + '_conv_1xk').get_weights()[0])

        if use_bias:
            bias.append(trained_model.get_layer(layer_name + '_conv_kxk').get_weights()[1])
            bias.append(trained_model.get_layer(layer_name + '_conv_kx1').get_weights()[1])
            bias.append(trained_model.get_layer(layer_name + '_conv_1xk').get_weights()[1])
        else:
            bias = [np.zeros((weights[0].shape[-1]),)]*3

        if use_bn:
            bn_kxk = trained_model.get_layer(layer_name + '_bn_kxk').get_weights()
            bn_kx1 = trained_model.get_layer(layer_name + '_bn_kx1').get_weights()
            bn_1xk = trained_model.get_layer(layer_name + '_bn_1xk').get_weights()

            gammas = [bn_kxk[0], bn_kx1[0], bn_1xk[0]]
            betas = [bn_kxk[1], bn_kx1[1], bn_1xk[1]]
            means = [bn_kxk[2], bn_kx1[2], bn_1xk[2]]
            vars = [bn_kxk[3], bn_kx1[3], bn_1xk[3]]
        else:
            gammas = [np.ones((weights[0].shape[-1]),)]*3
            betas = [np.zeros((weights[0].shape[-1]),)]*3
            means = [np.zeros((weights[0].shape[-1]),)]*3
            vars = [np.ones((weights[0].shape[-1]),)]*3

        kernel_size = weights[0].shape[0]

        w_kxk = (gammas[0] / np.sqrt(np.add(vars[0], 1e-10))) * weights[0]
        weights_kx1 = (gammas[1] / np.sqrt(np.add(vars[1], 1e-10))) * weights[1]
        weights_1xk = (gammas[2] / np.sqrt(np.add(vars[2], 1e-10))) * weights[2]

        bs = []
        for i in range(3):
            bs.append((((bias[i] - means[i]) * gammas[i]) / np.sqrt(np.add(vars[i], 1e-10))) + betas[i])
        w_k_1 = np.zeros_like(w_kxk)
        w_1_k = np.zeros_like(w_kxk)
        w_k_1[kernel_size // 2 - 1:kernel_size // 2 + 2, kernel_size // 2, :, :] = weights_kx1[:, 0, :, :]
        w_1_k[kernel_size // 2, kernel_size // 2 - 1:kernel_size // 2 + 2, :, :] = weights_1xk[0, :, :, :]
        weight = [w_kxk, w_k_1, w_1_k]
        bias = bs

        infer_model.get_layer(layer_name).set_weights(diff_model(model, weight, bias))


def fusion_rep_vgg(AC_names, trained_model, infer_model):
    """
                |
                |
    -------------------------
    |           |           |
    |           |           |
    1x1         kxk         |
    |           |           |
    |           |           |
    BN          BN          BN
    |           |           |
    |           |           |
    -----------combine-------
                |
                |
    RepVGG
    """
    for layer_name, use_bias, use_bn, model, epoch in AC_names:

        conv_kxk_weights = trained_model.get_layer(layer_name + '_conv_kxk').get_weights()[0]
        conv_1x1_weights = trained_model.get_layer(layer_name + '_conv_1x1').get_weights()[0]

        if use_bias:
            conv_kxk_bias = trained_model.get_layer(layer_name + '_conv_kxk').get_weights()[1]
            conv_1x1_bias = trained_model.get_layer(layer_name + '_conv_1x1').get_weights()[1]
        else:
            conv_kxk_bias = np.zeros((conv_kxk_weights.shape[-1],))
            conv_1x1_bias = np.zeros((conv_1x1_weights.shape[-1],))

        if use_bn:
            gammas_1x1, betas_1x1, means_1x1, var_1x1 = trained_model.get_layer(layer_name + '_bn_1').get_weights()
            gammas_kxk, betas_kxk, means_kxk, var_kxk = trained_model.get_layer(layer_name + '_bn_2').get_weights()
            gammas_res, betas_res, means_res, var_res = trained_model.get_layer(layer_name + '_bn_3').get_weights()

        else:
            gammas_1x1, betas_1x1, means_1x1, var_1x1 = [np.ones((conv_1x1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_weights.shape[-1],)),
                                                         np.ones((conv_1x1_weights.shape[-1],))]
            gammas_kxk, betas_kxk, means_kxk, var_kxk = [np.ones((conv_kxk_weights.shape[-1],)),
                                                         np.zeros((conv_kxk_weights.shape[-1],)),
                                                         np.zeros((conv_kxk_weights.shape[-1],)),
                                                         np.ones((conv_kxk_weights.shape[-1],))]
            gammas_res, betas_res, means_res, var_res = [np.ones((conv_1x1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_weights.shape[-1],)),
                                                         np.ones((conv_1x1_weights.shape[-1],))]

        w_kxk = (gammas_kxk / np.sqrt(np.add(var_kxk, 1e-10))) * conv_kxk_weights
        kernel_size = w_kxk.shape[0]
        in_channels = w_kxk.shape[2]
        w_1x1 = np.zeros_like(w_kxk)
        w_1x1[kernel_size // 2, kernel_size // 2, :, :] = (gammas_1x1 / np.sqrt(np.add(var_1x1, 1e-10))) * conv_1x1_weights
        w_res = np.zeros_like(w_kxk)

        for i in range(in_channels):
            w_res[kernel_size // 2, kernel_size // 2, i % in_channels, i] = 1
        w_res = ((gammas_res / np.sqrt(np.add(var_res, 1e-10))) * w_res)

        b_1x1 = (((conv_1x1_bias - means_1x1) * gammas_1x1) / np.sqrt(np.add(var_1x1, 1e-10))) + betas_1x1
        b_kxk = (((conv_kxk_bias - means_kxk) * gammas_kxk) / np.sqrt(np.add(var_kxk, 1e-10))) + betas_kxk
        b_res = (((0 - means_res) * gammas_res) / np.sqrt(np.add(var_res, 1e-10))) + betas_res

        weight = [w_res, w_1x1, w_kxk]
        bias = [b_res, b_1x1, b_kxk]

        infer_model.get_layer(layer_name).set_weights(diff_model(model, weight, bias))


def fusion_1x1_kxk_down(AC_names, trained_model, infer_model):
    """
                |
                |
    -------------------------
    |                       |
    |                       |
    1x1                     kxk
    |                       |
    |                       |
    BN                      BN
    |                       |
    |                       |
    -----------combine-------
                |
                |
    RepVGG
    """
    for layer_name, use_bias, use_bn, model, epoch in AC_names:

        conv_kxk_weights = trained_model.get_layer(layer_name + '_conv_kxk').get_weights()[0]
        conv_1x1_weights = trained_model.get_layer(layer_name + '_conv_1x1').get_weights()[0]

        if use_bias:
            conv_kxk_bias = trained_model.get_layer(layer_name + '_conv_kxk').get_weights()[1]
            conv_1x1_bias = trained_model.get_layer(layer_name + '_conv_1x1').get_weights()[1]
        else:
            conv_kxk_bias = np.zeros((conv_kxk_weights.shape[-1],))
            conv_1x1_bias = np.zeros((conv_1x1_weights.shape[-1],))

        if use_bn:
            gammas_1x1, betas_1x1, means_1x1, var_1x1 = trained_model.get_layer(layer_name + '_bn_1').get_weights()
            gammas_kxk, betas_kxk, means_kxk, var_kxk = trained_model.get_layer(layer_name + '_bn_2').get_weights()

        else:
            gammas_1x1, betas_1x1, means_1x1, var_1x1 = [np.ones((conv_1x1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_weights.shape[-1],)),
                                                         np.ones((conv_1x1_weights.shape[-1],))]
            gammas_kxk, betas_kxk, means_kxk, var_kxk = [np.ones((conv_kxk_weights.shape[-1],)),
                                                         np.zeros((conv_kxk_weights.shape[-1],)),
                                                         np.zeros((conv_kxk_weights.shape[-1],)),
                                                         np.ones((conv_kxk_weights.shape[-1],))]

        w_kxk = (gammas_kxk / np.sqrt(np.add(var_kxk, 1e-10))) * conv_kxk_weights
        kernel_size = w_kxk.shape[0]
        w_1x1 = np.zeros_like(w_kxk)
        w_1x1[kernel_size // 2, kernel_size // 2, :, :] = (gammas_1x1 / np.sqrt(np.add(var_1x1, 1e-10))) * conv_1x1_weights

        b_1x1 = (((conv_1x1_bias - means_1x1) * gammas_1x1) / np.sqrt(np.add(var_1x1, 1e-10))) + betas_1x1
        b_kxk = (((conv_kxk_bias - means_kxk) * gammas_kxk) / np.sqrt(np.add(var_kxk, 1e-10))) + betas_kxk

        weight = [w_1x1, w_kxk]
        bias = [b_1x1, b_kxk]

        infer_model.get_layer(layer_name).set_weights(diff_model(model, weight, bias))


def fusion_dbb(AC_names, trained_model, infer_model):
    """
                        |
                        |
    ---------------------------------------------
    |           |               |               |
    |           |               |               |
    1x1         1x1             1x1             kxk
    |           |               |               |
    |           |               |               |
    BN          BN              BN              BN
    |           |               |               |
    |           |               |               |
    |           kxk             avg             |
    |           |               |               |
    |           |               |               |
    |           BN              |               |
    --------------------Add----------------------
                        |
                        |
    Diverse Branch Block
    """
    for layer_name, use_bias, use_bn, model, epoch in AC_names:
        conv_1x1_1_weights = trained_model.get_layer(layer_name + '_conv_1x1_1').get_weights()[0]
        conv_1x1_2_weights = trained_model.get_layer(layer_name + '_conv_1x1_2').get_weights()[0]
        conv_1x1_3_weights = trained_model.get_layer(layer_name + '_conv_1x1_3').get_weights()[0]
        conv_kxk_1_weights = trained_model.get_layer(layer_name + '_conv_kxk_1').get_weights()[0]

        conv_kxk_2_weights = trained_model.get_layer(layer_name + '_conv_kxk_2').get_weights()[0]
        kernel_size = conv_kxk_2_weights.shape[0]
        # #
        in_channels = conv_kxk_2_weights.shape[2]
        conv_kxk_3_weights = np.zeros_like(conv_kxk_2_weights)
        for i in range(in_channels):
            conv_kxk_3_weights[:, :, i % in_channels, i] = 1.0 / (kernel_size*kernel_size)

        if use_bias:
            conv_kxk_1_bias = trained_model.get_layer(layer_name + '_conv_kxk_1').get_weights()[1]
            conv_kxk_2_bias = trained_model.get_layer(layer_name + '_conv_kxk_2').get_weights()[1]
            conv_1x1_1_bias = trained_model.get_layer(layer_name + '_conv_1x1_1').get_weights()[1]
            conv_1x1_2_bias = trained_model.get_layer(layer_name + '_conv_1x1_2').get_weights()[1]
            conv_1x1_3_bias = trained_model.get_layer(layer_name + '_conv_1x1_3').get_weights()[1]
        else:
            conv_kxk_1_bias = np.zeros((conv_kxk_1_weights.shape[-1],))
            conv_kxk_2_bias = np.zeros((conv_kxk_2_weights.shape[-1],))
            conv_1x1_1_bias = np.zeros((conv_1x1_1_weights.shape[-1],))
            conv_1x1_2_bias = np.zeros((conv_1x1_2_weights.shape[-1],))
            conv_1x1_3_bias = np.zeros((conv_1x1_3_weights.shape[-1],))
        conv_kxk_3_bias = np.zeros_like(conv_kxk_2_bias)

        if use_bn:
            gammas_1x1_1, betas_1x1_1, means_1x1_1, var_1x1_1 = trained_model.get_layer(layer_name + '_bn_1').get_weights()
            gammas_1x1_2_1, betas_1x1_2_1, means_1x1_2_1, var_1x1_2_1 = trained_model.get_layer(layer_name + '_bn_2_1').get_weights()
            gammas_kxk_2_2, betas_kxk_2_2, means_kxk_2_2, var_kxk_2_2 = trained_model.get_layer(layer_name + '_bn_2_2').get_weights()
            gammas_1x1_3_1, betas_1x1_3_1, means_1x1_3_1, var_1x1_3_1 = trained_model.get_layer(
                layer_name + '_bn_3_1').get_weights()
            gammas_kxk_3_2, betas_kxk_3_2, means_kxk_3_2, var_kxk_3_2 = trained_model.get_layer(
                layer_name + '_bn_3_2').get_weights()
            gammas_kxk_4, betas_kxk_4, means_kxk_4, var_kxk_4 = trained_model.get_layer(
                layer_name + '_bn_4').get_weights()


        else:
            gammas_1x1_1, betas_1x1_1, means_1x1_1, var_1x1_1 = [np.ones((conv_1x1_1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_1_weights.shape[-1],)),
                                                         np.ones((conv_1x1_1_weights.shape[-1],))]
            gammas_1x1_2_1, betas_1x1_2_1, means_1x1_2_1, var_1x1_2_1 = [np.ones((conv_1x1_2_weights.shape[-1],)),
                                                                 np.zeros((conv_1x1_1_weights.shape[-1],)),
                                                                 np.zeros((conv_1x1_1_weights.shape[-1],)),
                                                                 np.ones((conv_1x1_1_weights.shape[-1],))]

            gammas_1x1_3_1, betas_1x1_3_1, means_1x1_3_1, var_1x1_3_1 = [np.ones((conv_1x1_3_weights.shape[-1],)),
                                                                         np.zeros((conv_1x1_3_weights.shape[-1],)),
                                                                         np.zeros((conv_1x1_3_weights.shape[-1],)),
                                                                         np.ones((conv_1x1_3_weights.shape[-1],))]

            gammas_kxk_2_2, betas_kxk_2_2, means_kxk_2_2, var_kxk_2_2  = [np.ones((conv_kxk_2_weights.shape[-1],)),
                                                                          np.zeros((conv_kxk_2_weights.shape[-1],)),
                                                                          np.zeros((conv_kxk_2_weights.shape[-1],)),
                                                                          np.ones((conv_kxk_2_weights.shape[-1],))]

            gammas_kxk_3_2, betas_kxk_3_2, means_kxk_3_2, var_kxk_3_2 = [np.ones((conv_kxk_2_weights.shape[-1],)),
                                                                          np.zeros((conv_kxk_2_weights.shape[-1],)),
                                                                          np.zeros((conv_kxk_2_weights.shape[-1],)),
                                                                          np.ones((conv_kxk_2_weights.shape[-1],))]

            gammas_kxk_4, betas_kxk_4, means_kxk_4, var_kxk_4 = [np.ones((conv_kxk_1_weights.shape[-1],)),
                                                                 np.zeros((conv_kxk_1_weights.shape[-1],)),
                                                                 np.zeros((conv_kxk_1_weights.shape[-1],)),
                                                                 np.ones((conv_kxk_1_weights.shape[-1],))]

        w_1x1_2 = ((gammas_1x1_2_1 / np.sqrt(np.add(var_1x1_2_1, 1e-10))) * conv_1x1_2_weights).transpose([0, 1, 3, 2])
        w_kxk_2 = ((gammas_kxk_2_2 / np.sqrt(np.add(var_kxk_2_2, 1e-10))) * conv_kxk_2_weights).transpose([0, 1, 3, 2])

        b_1x1_2 = (((conv_1x1_2_bias - means_1x1_2_1) * gammas_1x1_2_1) / np.sqrt(np.add(var_1x1_2_1, 1e-10))) + betas_1x1_2_1
        b_kxk_2 = (((conv_kxk_2_bias - means_kxk_2_2) * gammas_kxk_2_2) / np.sqrt(np.add(var_kxk_2_2, 1e-10))) + betas_kxk_2_2

        w_1x1_3 = ((gammas_1x1_3_1 / np.sqrt(np.add(var_1x1_3_1, 1e-10))) * conv_1x1_3_weights).transpose([0, 1, 3, 2])
        w_kxk_3 = ((gammas_kxk_3_2 / np.sqrt(np.add(var_kxk_3_2, 1e-10))) * conv_kxk_3_weights).transpose([0, 1, 3, 2])

        b_1x1_3 = (((conv_1x1_3_bias - means_1x1_3_1) * gammas_1x1_3_1) / np.sqrt(np.add(var_1x1_3_1, 1e-10))) + betas_1x1_3_1
        b_kxk_3 = (((conv_kxk_3_bias - means_kxk_3_2) * gammas_kxk_3_2) / np.sqrt(np.add(var_kxk_3_2, 1e-10))) + betas_kxk_3_2

        with tf.Session() as sess:
            conv_1x1_2 = tf.convert_to_tensor(w_1x1_2.astype(np.float32))
            conv_kxk_2 = tf.convert_to_tensor(w_kxk_2.astype(np.float32))
            numpy_w_2 = K.conv2d(conv_kxk_2, conv_1x1_2, padding='same').eval()

            conv_1x1_3 = tf.convert_to_tensor(w_1x1_3.astype(np.float32))
            conv_kxk_3 = tf.convert_to_tensor(w_kxk_3.astype(np.float32))
            numpy_w_3 = K.conv2d(conv_kxk_3, conv_1x1_3, padding='same').eval()

        weight2 = numpy_w_2.transpose([0, 1, 3, 2])
        bias2 = np.sum(w_kxk_2 * b_1x1_2, axis=(0, 1, 3)) + b_kxk_2

        weight3 = numpy_w_3.transpose([0, 1, 3, 2])
        bias3 = np.sum(w_kxk_3 * b_1x1_3, axis=(0, 1, 3)) + b_kxk_3

        weight4 = (gammas_kxk_4 / np.sqrt(np.add(var_kxk_4, 1e-10))) * conv_kxk_1_weights
        weight1 = np.zeros_like(weight4)
        weight1[kernel_size // 2, kernel_size // 2, :, :] = (gammas_1x1_1 / np.sqrt(
            np.add(var_1x1_1, 1e-10))) * conv_1x1_1_weights

        bias1 = (((conv_1x1_1_bias - means_1x1_1) * gammas_1x1_1) / np.sqrt(np.add(var_1x1_1, 1e-10))) + betas_1x1_1
        bias4 = (((conv_kxk_1_bias - means_kxk_4) * gammas_kxk_4) / np.sqrt(np.add(var_kxk_4, 1e-10))) + betas_kxk_4

        weight = weight1 + weight2 + weight3 + weight4
        bias = bias1 + bias2 + bias3 + bias4

        # weight = weight1 + weight2 + weight4
        # bias = bias1 + bias2 + bias4
        infer_model.get_layer(layer_name).set_weights([weight, bias])


class DBB(object):
    def __init__(self, stage='train'):
        super(DBB, self).__init__()
        self.stage = stage
        self.dbb_block_names = {'dbb_same_kernel_size': [], 'dbb_dbb': [],
                                'dbb_asym': [], 'rep_vgg': [],
                                'dbb_diff_kernel_size': [], 'dbb_1x1_kxk_down': []}

    def rep_vgg(self, input, filters, kernel_size, name, dilation_rate=1,
                        use_bias=False, use_bn=True, model='Add', padding='same'):
        """
                    |
                    |
        -------------------------
        |           |           |
        |           |           |
        1x1         kxk         |
        |           |           |
        |           |           |
        BN          BN          BN
        |           |           |
        |           |           |
        -----------combine-------
                    |
                    |
        RepVGG
        """
        in_dim = int(input.shape[-1])
        assert in_dim == filters
        x = None
        if self.stage == 'train':
            conv_1x1 = Conv2D(filters, (1, 1), padding=padding, use_bias=use_bias,
                              dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_1x1')(input)

            conv_kxk = Conv2D(filters, (kernel_size, kernel_size), padding=padding, use_bias=use_bias,
                              dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_kxk')(input)

            if use_bn:
                conv_1x1 = BatchNormalization(name=name + '_bn_1')(conv_1x1)
                conv_kxk = BatchNormalization(name=name + '_bn_2')(conv_kxk)
                input = BatchNormalization(name=name + '_bn_3')(input)

            if model == 'Add':
                x = Add(name=name + '_add')([input, conv_1x1, conv_kxk])
            elif model == 'Concate':
                x = Concatenate(name=name + '_concate')([input, conv_1x1, conv_kxk])

        else:
            if model == 'Add':
                x = Conv2D(filters, kernel_size, dilation_rate=dilation_rate,
                               padding='same', name=name)(input)
            elif model == 'Concate':
                x = Conv2D(3*filters, kernel_size, dilation_rate=dilation_rate,
                           padding='same', name=name)(input)
            self.dbb_block_names['rep_vgg'].append([name, use_bias, use_bn, model, None])

        return x

    def dbb_diff_kernel_size(self, input, filters, name, dilation_rate=1, use_bias=False, use_bn=True, epoch=2, model='Add', padding='same'):
        """
                    |
                    |
        -----------------------------
        |           |               |
        |           |               |
        1x1         3x3             5x5
        |           |               |
        |           |               |
        BN          ...             BN
        |           |               |
        |           |               |
        ------------Add--------------
                    |
                    |
        Diverse Branch Block
        """
        assert epoch > 2
        x = None
        if self.stage == 'train':
            xs = []
            for k in range(epoch):
                x = Conv2D(filters, 2*k + 1, padding=padding, use_bias=use_bias,
                           dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_%d' % k)(input)
                if use_bn:
                    x = BatchNormalization(name=name + '_bn_%d' % k)(x)
                xs.append(x)
            x = Add(name=name + '_add')(xs)

        else:

            x = Conv2D(filters, 2*(epoch-1) + 1, dilation_rate=dilation_rate,
                           padding='same', name=name)(input)

            self.dbb_block_names['dbb_diff_kernel_size'].append([name, use_bias, use_bn, None, epoch])
        return x

    def dbb_same_kernel_size(self, input, filters, kernel_size, name, dilation_rate=1, use_bias=False, use_bn=True, epoch=2, model='Add', padding='same'):
        """
                    |
                    |
        -----------------------------
        |           |               |
        |           |               |
        kxk         ...             kxk
        |           |               |
        |           |               |
        BN          ...             BN
        |           |               |
        |           |               |
        -----------combine-----------
                    |
                    |
        Diverse Branch Block
        """
        assert epoch > 2
        x = None
        if self.stage == 'train':
            xs = []
            for k in range(epoch):
                x = Conv2D(filters, (kernel_size, kernel_size), padding=padding, use_bias=use_bias,
                           dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_%d' % k)(input)
                if use_bn:
                    x = BatchNormalization(name=name + '_bn_%d' % k)(x)
                xs.append(x)

            if model == 'Add':
                x = Add(name=name + '_add')(xs)
            elif model == 'Concate':
                x = Concatenate(name=name + '_concate')(xs)
        else:
            if model == 'Add':
                x = Conv2D(filters, kernel_size, dilation_rate=dilation_rate,
                           padding='same', name=name)(input)
            elif model == 'Concate':
                x = Conv2D(filters*epoch, kernel_size, dilation_rate=dilation_rate,
                           padding='same', name=name)(input)
            self.dbb_block_names['dbb_same_kernel_size'].append([name, use_bias, use_bn, model, epoch])
        return x

    def dbb_asym(self, input, filters, kernel_size, name, dilation_rate=1,
                        use_bias=False, use_bn=True, model='Add', padding='same'):
        """
                    |
                    |
        -----------------------------
        |           |               |
        |           |               |
        1xk         kx1             kxk
        |           |               |
        |           |               |
        BN          BN             BN
        |           |               |
        |           |               |
        -----------combine-----------
                    |
                    |
        Diverse Branch Block
        """
        x = None
        if self.stage == 'train':
            conv_kxk = Conv2D(filters, (kernel_size, kernel_size), padding=padding,
                              dilation_rate=(dilation_rate, dilation_rate), use_bias=use_bias,
                              name=name + '_conv_kxk')(input)
            conv_kx1 = Conv2D(filters, (kernel_size, 1), padding=padding,
                              dilation_rate=(dilation_rate, 1), use_bias=use_bias,
                              name=name + '_conv_kx1')(input)

            conv_1xk = Conv2D(filters, (1, kernel_size), padding=padding,
                              dilation_rate=(1, dilation_rate), use_bias=use_bias,
                              name=name + '_conv_1xk')(input)
            if use_bn:
                conv_kxk = BatchNormalization(axis=-1, name=name + '_bn_kxk')(conv_kxk)
                conv_kx1 = BatchNormalization(axis=-1, name=name + '_bn_kx1')(conv_kx1)
                conv_1xk = BatchNormalization(axis=-1, name=name + '_bn_1xk')(conv_1xk)

            if model == 'Add':
                x = Add(name=name + '_add')([conv_kxk, conv_kx1, conv_1xk])
            elif model == 'Concate':
                x = Concatenate(name=name + '_concate')([conv_kxk, conv_kx1, conv_1xk])
        else:
            if model == 'Add':
                x = Conv2D(filters, kernel_size, dilation_rate=dilation_rate,
                           padding='same', name=name)(input)
            elif model == 'Concate':
                x = Conv2D(filters * 3, kernel_size, dilation_rate=dilation_rate,
                           padding='same', name=name)(input)
            self.dbb_block_names['dbb_asym'].append([name, use_bias, use_bn, model, None])

        return x

    def dbb_dbb(self, input, filters, kernel_size, name, dilation_rate=1, use_bias=False, use_bn=True, padding='same'):
        """
                            |
                            |
        ---------------------------------------------
        |           |               |               |
        |           |               |               |
        1x1         1x1             1x1             kxk
        |           |               |               |
        |           |               |               |
        BN          BN              BN              BN
        |           |               |               |
        |           |               |               |
        |           kxk             avg             |
        |           |               |               |
        |           |               |               |
        |           BN              |               |
        --------------------Add----------------------
                            |
                            |
        Diverse Branch Block
        """
        x = None
        if self.stage == 'train':
            x1 = Conv2D(filters, 1, padding=padding, use_bias=use_bias,
                        dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_1x1_1')(input)
            x2 = Conv2D(filters, 1, padding=padding, use_bias=use_bias,
                        dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_1x1_2')(input)
            x3 = Conv2D(filters, 1, padding=padding, use_bias=use_bias,
                        dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_1x1_3')(input)
            x4 = Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                        dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_kxk_1')(input)
            if use_bn:
                x2 = BatchNormalization(name=name + '_bn_2_1')(x2)
                x3 = BatchNormalization(name=name + '_bn_3_1')(x3)

            x2 = Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                       dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_kxk_2')(x2)

            x3 = AveragePooling2D(kernel_size, strides=1, padding='same', name=name + '_avg')(x3)

            # avg_weights = np.zeros(shape=(kernel_size, kernel_size, filters, filters))
            # for i in range(filters):
            #     avg_weights[:, :, i % filters, i] = 1. / (kernel_size * kernel_size)
            # avg = Conv2D(filters, kernel_size, padding=padding, use_bias=False,
            #            dilation_rate=(dilation_rate, dilation_rate), name=name + '_avg')
            # x3 = avg(x3)
            # avg.set_weights([avg_weights])
            # avg.trainable = False

            if use_bn:
                x1 = BatchNormalization(name=name + '_bn_1')(x1)
                x2 = BatchNormalization(name=name + '_bn_2_2')(x2)
                x3 = BatchNormalization(name=name + '_bn_3_2')(x3)
                x4 = BatchNormalization(name=name + '_bn_4')(x4)

            x = Add(name=name + '_add')([x1, x2, x3, x4])
        else:
            x = Conv2D(filters, kernel_size, dilation_rate=dilation_rate,
                       padding='same', name=name)(input)

            self.dbb_block_names['dbb_dbb'].append([name, use_bias, use_bn, None, None])
        return x

    def dbb_1x1_kxk_down(self, input, filters, kernel_size, name, strides=2,
                        use_bias=False, use_bn=True, model='Add',
                        padding='same'):
        """
                    |
                    |
        -------------------------
        |                       |
        |                       |
        1x1                     kxk
        |                       |
        |                       |
        BN                      BN
        |                       |
        |                       |
        -----------combine-------
                    |
                    |
        RepVGG
        """

        x = None
        if self.stage == 'train':
            conv_1x1 = Conv2D(filters, (1, 1), padding=padding, use_bias=use_bias,
                              strides=strides, name=name + '_conv_1x1')(input)

            conv_kxk = Conv2D(filters, (kernel_size, kernel_size), padding=padding, use_bias=use_bias,
                              strides=strides, name=name + '_conv_kxk')(input)

            if use_bn:
                conv_1x1 = BatchNormalization(name=name + '_bn_1')(conv_1x1)
                conv_kxk = BatchNormalization(name=name + '_bn_2')(conv_kxk)

            if model == 'Add':
                x = Add(name=name + '_add')([conv_1x1, conv_kxk])
            elif model == 'Concate':
                x = Concatenate(axis=-1, name=name + '_concate')([conv_1x1, conv_kxk])
        else:
            if model == 'Add':
                x = Conv2D(filters, kernel_size, strides=strides,
                           padding='same', name=name)(input)
            elif model == 'Concate':
                x = Conv2D(filters * 2, kernel_size, strides=strides,
                           padding='same', name=name)(input)

            self.dbb_block_names['dbb_1x1_kxk_down'].append([name, use_bias, use_bn, model, None])

        return x

    def dbb_1x1_kxk(self, input, filters, kernel_size, name, dilation_rate=1,
                    use_bias=False, use_bn=True, padding='same'):
        """

                   |
                   |
                 1x1
                   |
                   |
                  kxk
                   |
                   |

        Diverse Branch Block
        """
        x = None
        if self.stage == 'train':
            x = Conv2D(filters, (1, 1), padding=padding, use_bias=use_bias,
                       dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_1x1')(input)
            if use_bn:
                x = BatchNormalization(name=name + '_bn_1')(x)
            x = Conv2D(filters, (kernel_size, kernel_size), padding=padding, use_bias=use_bias,
                       dilation_rate=(dilation_rate, dilation_rate), name=name + '_conv_kxk')(x)
            if use_bn:
                x = BatchNormalization(name=name + '_bn_2')(x)
        else:

            x = Conv2D(filters, kernel_size, dilation_rate=dilation_rate,
                       padding='same', name=name)(input)
            self.dbb_block_names['dbb_1x1_kxk'].append([name, use_bias, use_bn, None, None])
        return x

    def fusion(self, model, model_infer):
        dbb_same_kernel_size_names = self.dbb_block_names['dbb_same_kernel_size']
        fusion_same_kernel_size(dbb_same_kernel_size_names, model, model_infer)

        dbb_diff_kernel_size_names = self.dbb_block_names['dbb_diff_kernel_size']
        fusion_diff_kernel_size(dbb_diff_kernel_size_names, model, model_infer)

        dbb_asym_names = self.dbb_block_names['dbb_asym']
        fusion_asym(dbb_asym_names, model, model_infer)

        dbb_dbb_names = self.dbb_block_names['dbb_dbb']
        fusion_dbb(dbb_dbb_names, model, model_infer)

        rep_vgg_names = self.dbb_block_names['rep_vgg']
        fusion_rep_vgg(rep_vgg_names, model, model_infer)

        dbb_1x1_kxk_down_names = self.dbb_block_names['dbb_1x1_kxk_down']
        fusion_1x1_kxk_down(dbb_1x1_kxk_down_names, model, model_infer)
