#!/usr/bin/env python
'''
Script for predicting PHOCs for a number of images residing in a folder on
disk.
'''
import argparse
import logging
import os

import caffe
import numpy as np
import cv2

from phocnet.evaluation.cnn import net_output_for_word_image_list


def load_siamese_model(siamese_model, siamese_proto, phoc_proto):
    siamese_model = caffe.Net(siamese_proto, siamese_model, caffe.TEST)
    phoc_model = caffe.Net(phoc_proto, caffe.TEST)

    # ignore last layer
    layers = [f for f in phoc_model.params.keys() if f != 'fc8']
    for layer in layers:
        # for the Siamese network, load only one branch as PHOCNet base
        pre_layer = '{}l'.format(layer)
        phoc_model.params[layer][0].data[...] = (siamese_model
                                                 .params[pre_layer][0].data)
        phoc_model.params[layer][1].data[...] = (siamese_model
                                                 .params[pre_layer][1].data)

    return phoc_model


def load_images(img_dir):
    # check input structure: plain vs. folder structure
    img_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    if len(img_names) == 0:
        classes = [f for f in os.listdir(img_dir)
                   if os.path.isdir(os.path.join(img_dir, f))]
        img_paths = [os.path.join(img_dir, c, f)
                     for c in classes
                     for f in os.listdir(os.path.join(img_dir, c))
                     if f.endswith('.png')]
        img_ids = ['{}_{}'.format(os.path.basename(os.path.dirname(f)),
                                  os.path.basename(f).split('.')[0])
                   for f in img_paths]
    else:
        img_ids = [f.split('.')[0] for f in img_names]
        img_paths = [os.path.join(img_dir, f) for f in img_names]

    imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in img_paths]

    return imgs, img_ids


def main(img_dir, output_dir, pretrained_phocnet, deploy_proto,
         output_layer, min_image_width_height, gpu_id):
    logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=logging_format)
    logger = logging.getLogger('Predict PHOCs')

    if gpu_id is None:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

    logger.info('Loading PHOCNet...')
    phocnet = caffe.Net(deploy_proto, caffe.TEST,
                        weights=pretrained_phocnet)

    # find all images in the supplied dir
    word_img_list, img_ids = load_images(img_dir)
    logger.info('Found %d word images to process', len(img_ids))
    # push images through the PHOCNet
    logger.info('Predicting PHOCs...')
    predicted_phocs = net_output_for_word_image_list(
                                phocnet=phocnet,
                                word_img_list=word_img_list,
                                min_img_width_height=min_image_width_height,
                                output_layer=output_layer)
    # save everything
    logger.info('Saving...')
    np.savez(os.path.join(output_dir,
                          'predicted_output_{}.npz'.format(output_layer)),
             img_ids=img_ids,
             output=predicted_phocs)
    logger.info('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict PHOCs from a '
                                     'pretrained PHOCNet. The PHOCs are saved '
                                     ' as Numpy Array to disk.')
    parser.add_argument('--min_image_width_height', '-miwh', action='store',
                        type=int, default=26, help='The minimum image width '
                        'or height to be passed through the PHOCNet. '
                        'Default: 26')
    parser.add_argument('--output_dir', '-od', action='store', type=str,
                        default='.', help='The directory where to store the '
                        'PHOC Numpy Array. Default: .')
    parser.add_argument('--img_dir', '-id', action='store', type=str,
                        required=True, help='All images in this folder are '
                        'processed in ASCII order of their respective names.'
                        ' A PHOC is predicted for each.')
    parser.add_argument('--pretrained_phocnet', '-pp', action='store',
                        type=str, required=True, help='Path to a pretrained '
                        'PHOCNet binaryproto file.')
    parser.add_argument('--deploy_proto', '-dp', action='store', type=str,
                        required=True, help='Path to PHOCNet deploy prototxt '
                        'file.')
    parser.add_argument('--output_layer', help='Store output of provided '
                        'layer. Default: sigmoid', default='sigmoid')
    parser.add_argument('--gpu_id', '-gpu', action='store', type=int,
                        help='The ID of the GPU to use. If not specified, '
                        'training is run in CPU mode.')

    args = vars(parser.parse_args())
    main(**args)
