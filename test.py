import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
from Model import InpaintCAModel

test_image_folder = '/media/HardDisk/zhangchunping/image_inpainting/1125/test_result/input_all/input_all/'
test_image_list = os.listdir(test_image_folder)
mask_image = '/media/HardDisk/zhangchunping/image_inpainting/1125/examples/center_mask_256.png'
output_image_folder = './result/'
checkpoint_dir = '/media/HardDisk/zhangchunping/image_inpainting/1125/model_logs/20200604004311315382_ubuntu_celeba_NORMAL_wgan_gp_full_model_celeba_256/'

if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder)



if __name__ == "__main__":
#    ng.get_gpus(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = InpaintCAModel()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    # tf.reset_default_graph()
    input_image_tf = tf.placeholder(tf.float32, shape=(1,256,512,3))
    output = model.build_server_graph(input_image_tf)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    with tf.Session(config=sess_config) as sess:

        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        # load input image
        for test_image_name in test_image_list:
            test_image = os.path.join(test_image_folder, test_image_name)
            output_image = os.path.join(output_image_folder, test_image_name)
            print(output_image)
            image = cv2.imread(test_image)
            mask = cv2.imread(mask_image)

            assert image.shape == mask.shape

            # h, w, _ = image.shape
            # grid = 8
            # image = image[:h // grid * grid, :w // grid * grid, :]
            # mask = mask[:h // grid * grid, :w // grid * grid, :]
            print('Shape of image: {}'.format(image.shape))

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)


            result = sess.run(output,feed_dict={input_image_tf: input_image})
            cv2.imwrite(output_image, result[0][:, :, ::-1])
