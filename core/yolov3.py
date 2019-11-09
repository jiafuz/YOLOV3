import tensorflow as tf
import numpy as np


class bn_relu(tf.keras.Model):
    def __init__(self):
        super(bn_relu, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, input_data):
        output_data = self.bn(input_data)
        output_data = self.relu(output_data)
        return output_data


class residual_block(tf.keras.Model):
    def __init__(self, first_out_ch, second_out_ch):
        super(residual_block, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=first_out_ch, kernel_size=(1, 1), padding="same")
        self.bn_relu1 = bn_relu()
        self.conv2 = tf.keras.layers.Conv2D(filters=second_out_ch, kernel_size=(3, 3), padding="same")
        self.bn_relu2 = bn_relu()

    def call(self, input_data):
        output_data = self.conv1(input_data)
        output_data = self.bn_relu1(output_data)
        output_data = self.conv2(output_data)
        output_data = self.bn_relu2(output_data)
        result = input_data + output_data
        return result


class conv_bn_relu_maxpooling(tf.keras.Model):
    def __init__(self, out_ch):
        super(conv_bn_relu_maxpooling, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=(3, 3), padding="same")
        self.bn_relu = bn_relu()
        self.maxpooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, input_data):
        output_data = self.conv(input_data)
        output_data = self.bn_relu(output_data)
        output_data = self.maxpooling(output_data)
        return output_data


class darknet53(tf.keras.Model):
    def __init__(self):
        super(darknet53, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")
        self.bn_relu1 = bn_relu()
        self.conv_bn_relu_maxpooling1 = conv_bn_relu_maxpooling(64)
        # 1
        self.residual_block1 = residual_block(first_out_ch=32, second_out_ch=64)
        self.conv_bn_relu_maxpooling2 = conv_bn_relu_maxpooling(128)
        # 2
        self.residual_block2 = residual_block(first_out_ch=64, second_out_ch=128)
        self.residual_block3 = residual_block(first_out_ch=64, second_out_ch=128)
        self.conv_bn_relu_maxpooling3 = conv_bn_relu_maxpooling(256)
        # 8
        self.residual_block4 = residual_block(first_out_ch=128, second_out_ch=256)
        self.residual_block5 = residual_block(first_out_ch=128, second_out_ch=256)
        self.residual_block6 = residual_block(first_out_ch=128, second_out_ch=256)
        self.residual_block7 = residual_block(first_out_ch=128, second_out_ch=256)
        self.residual_block8 = residual_block(first_out_ch=128, second_out_ch=256)
        self.residual_block9 = residual_block(first_out_ch=128, second_out_ch=256)
        self.residual_block10 = residual_block(first_out_ch=128, second_out_ch=256)
        self.residual_block11 = residual_block(first_out_ch=128, second_out_ch=256)
        self.conv_bn_relu_maxpooling4 = conv_bn_relu_maxpooling(512)
        # 8
        self.residual_block12 = residual_block(first_out_ch=256, second_out_ch=512)
        self.residual_block13 = residual_block(first_out_ch=256, second_out_ch=512)
        self.residual_block14 = residual_block(first_out_ch=256, second_out_ch=512)
        self.residual_block15 = residual_block(first_out_ch=256, second_out_ch=512)
        self.residual_block16 = residual_block(first_out_ch=256, second_out_ch=512)
        self.residual_block17 = residual_block(first_out_ch=256, second_out_ch=512)
        self.residual_block18 = residual_block(first_out_ch=256, second_out_ch=512)
        self.residual_block19 = residual_block(first_out_ch=256, second_out_ch=512)
        self.conv_bn_relu_maxpooling5 = conv_bn_relu_maxpooling(1024)
        # 4
        self.residual_block20 = residual_block(first_out_ch=512, second_out_ch=1024)
        self.residual_block21 = residual_block(first_out_ch=512, second_out_ch=1024)
        self.residual_block22 = residual_block(first_out_ch=512, second_out_ch=1024)
        self.residual_block23 = residual_block(first_out_ch=512, second_out_ch=1024)

    def call(self, input_data):
        output_data = self.conv1(input_data)
        output_data = self.bn_relu1(output_data)
        output_data = self.conv_bn_relu_maxpooling1(output_data)
        # 1
        output_data = self.residual_block1(output_data)
        output_data = self.conv_bn_relu_maxpooling2(output_data)
        # 2
        output_data = self.residual_block2(output_data)
        output_data = self.residual_block3(output_data)
        output_data = self.conv_bn_relu_maxpooling3(output_data)
        # 8
        output_data = self.residual_block4(output_data)
        output_data = self.residual_block5(output_data)
        output_data = self.residual_block6(output_data)
        output_data = self.residual_block7(output_data)
        output_data = self.residual_block8(output_data)
        output_data = self.residual_block9(output_data)
        output_data = self.residual_block10(output_data)
        output_data = self.residual_block11(output_data)
        rout1 = output_data
        output_data = self.conv_bn_relu_maxpooling4(output_data)
        # 8
        output_data = self.residual_block12(output_data)
        output_data = self.residual_block13(output_data)
        output_data = self.residual_block14(output_data)
        output_data = self.residual_block15(output_data)
        output_data = self.residual_block16(output_data)
        output_data = self.residual_block17(output_data)
        output_data = self.residual_block18(output_data)
        output_data = self.residual_block19(output_data)
        rout2 = output_data
        output_data = self.conv_bn_relu_maxpooling5(output_data)
        # 4
        output_data = self.residual_block20(output_data)
        output_data = self.residual_block21(output_data)
        output_data = self.residual_block22(output_data)
        output_data = self.residual_block23(output_data)
        rout3 = output_data

        return rout1, rout2, rout3


class convolutional_set(tf.keras.Model):
    def __init__(self, out_ch):
        super(convolutional_set, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=(1, 1), padding="same")
        self.bn_relu1 = bn_relu()
        self.conv2 = tf.keras.layers.Conv2D(filters=out_ch * 2, kernel_size=(3, 3), padding="same")
        self.bn_relu2 = bn_relu()
        self.conv3 = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=(1, 1), padding="same")
        self.bn_relu3 = bn_relu()
        self.conv4 = tf.keras.layers.Conv2D(filters=out_ch * 2, kernel_size=(3, 3), padding="same")
        self.bn_relu4 = bn_relu()
        self.conv5 = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=(1, 1), padding="same")
        self.bn_relu5 = bn_relu()

    def call(self, input_data):
        output_data = self.conv1(input_data)
        output_data = self.bn_relu1(output_data)
        output_data = self.conv2(output_data)
        output_data = self.bn_relu2(output_data)
        output_data = self.conv3(output_data)
        output_data = self.bn_relu3(output_data)
        output_data = self.conv4(output_data)
        output_data = self.bn_relu4(output_data)
        output_data = self.conv5(output_data)
        output_data = self.bn_relu5(output_data)

        return output_data


class YOLOV3(tf.keras.Model):
    def __init__(self, num_class):
        super(YOLOV3, self).__init__()
        self.darknet53 = darknet53()
        self.convolutional_set1 = convolutional_set(512)
        self.conv1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="same")
        self.bn_relu1 = bn_relu()
        self.conv_Lbbox = tf.keras.layers.Conv2D(filters=3 * (num_class + 5), kernel_size=(1, 1), padding="same")

        self.conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.convolutional_set2 = convolutional_set(256)
        self.conv3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")
        self.bn_relu2 = bn_relu()
        self.conv_Mbbox = tf.keras.layers.Conv2D(filters=3 * (num_class + 5), kernel_size=(1, 1), padding="same")

        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.convolutional_set3 = convolutional_set(128)
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")
        self.bn_relu3 = bn_relu()
        self.conv_Sbbox = tf.keras.layers.Conv2D(filters=3 * (num_class + 5), kernel_size=(1, 1), padding="same")

    def call(self, input_data):
        small_recept_field, medium_recept_field, large_recept_field = self.darknet53(input_data)
        conv_set1 = self.convolutional_set1(large_recept_field)
        conv_Lbbox = self.conv1(conv_set1)
        conv_Lbbox = self.bn_relu1(conv_Lbbox)
        predicted_conv_Lbbox = self.conv_Lbbox(conv_Lbbox)

        conv_Mbbox = self.conv2(conv_set1)
        conv_Mbbox = tf.image.resize(conv_Mbbox, (26, 26), method="nearest")
        conv_Mbbox = tf.concat([conv_Mbbox, medium_recept_field], axis=-1)
        conv_Mbbox = self.convolutional_set2(conv_Mbbox)
        conv_Mbbox = self.conv3(conv_Mbbox)
        conv_Mbbox = self.bn_relu2(conv_Mbbox)
        predicted_conv_Mbbox = self.conv_Mbbox(conv_Mbbox)

        conv_Sbbox = self.conv4(conv_set1)
        conv_Sbbox = tf.image.resize(conv_Sbbox, (52, 52), method="nearest")
        conv_Sbbox = tf.concat([conv_Sbbox, small_recept_field], axis=-1)
        conv_Sbbox = self.convolutional_set3(conv_Sbbox)
        conv_Sbbox = self.conv5(conv_Sbbox)
        conv_Sbbox = self.bn_relu3(conv_Sbbox)
        predicted_conv_Sbbox = self.conv_Sbbox(conv_Sbbox)

        return predicted_conv_Lbbox, predicted_conv_Mbbox, predicted_conv_Sbbox

if __name__=="__main__":
    model = YOLOV3(2)
    # model.call(input_data=tf.random.uniform(shape=(1,416,416,3)))
    input_data = tf.random.uniform(shape=(2, 416, 416, 3))
    output = model(input_data)
    print(np.shape(output[0]))
    print(np.shape(output[1]))
    print(np.shape(output[2]))
    # print(tf.image.resize(input_data,(400,400),method="nearest"))