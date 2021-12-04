import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from skimage.transform import rescale, resize
from time import time
import argparse
import ast

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # GPU number, -1 for CPU
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list = []
    no_face_image = np.zeros(200000)
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)
    j = 1
    for i, image_path in enumerate(image_path_list):
        name = image_path.strip().split('/')[-1][:-4]
        image = imread(image_path)

        if args.isDlib:
            import dlib
            detector_path = os.path.join('.', 'Data/net-data/mmod_human_face_detector.dat')
            face_detector = dlib.cnn_face_detection_model_v1(
                    detector_path)
        detected_faces = face_detector(image,1)
        if len(detected_faces) == 0:

            no_face_image[j] = name
            no_face_image_path = os.path.join(save_folder,'no_face_images.txt' )
            if not os.path.exists(no_face_image_path):
                os.mkdir(no_face_image_path)
            np.savetxt("no_face_images.txt",  no_face_image)
            print('warning: no detected face')
            j = j+1
            continue
        if len(detected_faces) != 0:
            d = detected_faces[0].rect  ## only use the first detected face (assume that each input image only contains one face)
            left = d.left();
            right = d.right();
            top = d.top();
            bottom = d.bottom()
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
            size = int(old_size * 1.58)

            # crop image
            src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
            DST_PTS = np.array([[0, 0], [0, 256 - 1], [256 - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            print(tform.inverse)
            image = image / 255.
            cropped_image = warp(image, tform.inverse, output_shape=(256, 256))
            image_path = os.path.join(save_folder, name + '_cropped.jpg')
            imsave(image_path, cropped_image)
            print(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cropped_image')

    parser.add_argument('-i', '--inputDir', default='datas', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='./datas/out', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if Fal')
    main(parser.parse_args())
