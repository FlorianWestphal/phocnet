import os
import numpy as np
import io
import cv2
import re
import random

from phocnet.caffe.lmdb_creator import CaffeLMDBCreator
from phocnet.attributes.phoc_creator import PHOCCreator
from phocnet.caffe.augmentation import AugmentationCreator


# min width-height is 26 for SPP, but 30 for TPP layer
MIN_IMG_WIDTH_HEIGHT = 30
CSV_PATTERN = '([0-9]+),(.*)'
IIIT_PATTERN = '(.*) (.*) ([0-9]+) ([01]+)'
RANDOM_LIMITS = (0.8, 1.1)
TRANSFORM = 'transform'
RESIZE = 'resize'


def is_csv_file(label_list):
    label_match = re.compile(CSV_PATTERN)
    # check file format of label list
    with io.open(label_list, 'r', encoding='utf-8') as f:
        line = f.readline()
        if label_match.match(line) is not None:
            label_csv = True
        else:
            label_csv = False

    return label_csv


def group_by_id(img_dir):
    files = {}

    imgs = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    if len(imgs) == 0:
        ids = [f for f in os.listdir(img_dir)
               if os.path.isdir(os.path.join(img_dir, f))]
        for word_id in ids:
            files[word_id] = []
            path = os.path.join(img_dir, word_id)
            for img in os.listdir(path):
                if not img.endswith('.png'):
                    continue
                files[word_id].append(os.path.join(word_id, img))
    else:
        for img in imgs:
            word_id = img.split('_')[0]
            if word_id not in files:
                files[word_id] = []
            files[word_id].append(img)

    return files


def read_label_csv(img_dir, label_list, upper):
    label_img_map = {}
    size = 0

    label_match = re.compile(CSV_PATTERN)
    files = group_by_id(img_dir)

    with io.open(label_list, 'r', encoding='utf-8') as f:
        for line in f:
            match = label_match.match(line)
            word = match.group(2).strip()
            if upper:
                word = word.upper()
            word_id = match.group(1)
            if word_id in files:
                label_img_map[word] = files[word_id]
                size += len(files[word_id])

    return label_img_map, size


def get_word_list(label_list, upper, test=None):
    words = []

    if is_csv_file(label_list):
        label_match = re.compile(CSV_PATTERN)
        with io.open(label_list, 'r', encoding='utf-8') as f:
            for line in f:
                match = label_match.match(line)
                word = match.group(2).strip()
                if upper:
                    word = word.upper()
                words.append(word)
    else:
        label_img_map, _ = read_label_iiit(label_list, test)
        words = list(label_img_map.keys())

    return words


def read_label_iiit(label_list, test):
    label_img_map = {}
    size = 0

    add = '1' if test else '0'
    label_match = re.compile(IIIT_PATTERN)
    with io.open(label_list, 'r', encoding='utf-8') as f:
        for line in f:
            match = label_match.match(line)
            train_test = match.group(4)
            word = match.group(2).upper()
            path = match.group(1)
            if train_test == add:
                if word not in label_img_map:
                    label_img_map[word] = []
                label_img_map[word].append(path)
                size += 1

    return label_img_map, size


def map_labels_to_imgs(img_dir, label_list, test, upper):

    if is_csv_file(label_list):
        label_img_map, size = read_label_csv(img_dir, label_list, upper)
    else:
        label_img_map, size = read_label_iiit(label_list, test)

    return label_img_map, size


def setup_lmdb(output_dir, dataset_name, test):
    mode = 'test' if test else 'train'
    word_name = ('{}_nti500000_pul2-3-4-5_'
                 '{}_word_images_lmdb'.format(dataset_name, mode))
    phoc_name = ('{}_nti500000_pul2-3-4-5_'
                 '{}_phocs_lmdb'.format(dataset_name, mode))
    train_word_images_lmdb_path = os.path.join(output_dir, word_name)
    train_phoc_lmdb_path = os.path.join(output_dir, phoc_name)

    lmdb_creator = CaffeLMDBCreator()
    lmdb_creator.open_dual_lmdb_for_write(
                                image_lmdb_path=train_word_images_lmdb_path,
                                additional_lmdb_path=train_phoc_lmdb_path,
                                create=True)
    return lmdb_creator


def add_img(img, phoc, label, label_id, lmdb_creator, random_indices):
    pseudo_label = random_indices.pop()
    # add random index as start of the key to enforce random reading
    # order from LMDB
    key = '{}_{}'.format(str(pseudo_label).zfill(8),
                         label.encode('ascii', 'ignore'))
    lmdb_creator.put_dual_adjust(img, phoc, label_id,
                                 MIN_IMG_WIDTH_HEIGHT, key)


def random_resize(img):
    scale = random.random() + 1

    return cv2.resize(img, (int(img.shape[0] * scale),
                            int(img.shape[1] * scale)))


def label_id_from_name(name):
    parts = name.split('/')
    if len(parts) > 1:
        return int(parts[0])
    else:
        return int(name.split('_')[0])


def match_computed_phocs(img_dir, label_img_map, phocs, lmdb_creator,
                         size, n_train_images, augment):

    size = size if n_train_images is None else n_train_images
    random_indices = range(size)
    np.random.shuffle(random_indices)

    for label in label_img_map:
        phoc = phocs[label]
        if n_train_images is None:
            for img_name in label_img_map[label]:
                img = cv2.imread(os.path.join(img_dir, img_name),
                                 cv2.IMREAD_GRAYSCALE)
                label_id = label_id_from_name(img_name)
                add_img(img, phoc, label, label_id, lmdb_creator,
                        random_indices)
        else:
            imgs_per_label = size / len(label_img_map)
            paths = list(label_img_map[label])
            imgs = []
            if len(paths) > imgs_per_label:
                random.shuffle(paths)
                paths = paths[:imgs_per_label]
            for path in paths:
                img = cv2.imread(os.path.join(img_dir, path),
                                 cv2.IMREAD_GRAYSCALE)
                imgs.append(img)
                if augment == RESIZE:
                    img = random_resize(img)
                label_id = label_id_from_name(path)
                add_img(img, phoc, label, label_id, lmdb_creator,
                        random_indices)

            if augment == TRANSFORM and len(imgs) < imgs_per_label:
                idxs = np.random.randint(len(imgs),
                                         size=imgs_per_label - len(imgs))
                for idx in idxs:
                    aug_img = (AugmentationCreator
                               .create_affine_transform_augmentation(
                                   img=imgs[idx], random_limits=RANDOM_LIMITS))
                    add_img(aug_img, phoc, label, label_id, lmdb_creator,
                            random_indices)


def compute_phocs(labels, phoc_unigram_levels, word_list, warn):
    phocs = {}

    word_list = word_list if word_list is not None else labels
    phoc_creator = PHOCCreator(word_list, phoc_unigram_levels, warn=warn)
    for label in labels:
        phocs[label] = phoc_creator.create_phoc(label)

    return phocs


def main(img_dir, label_list, output_dir, dataset_name,
         phoc_unigram_levels, n_train_images, vocabulary, test, upper,
         augment):
    label_img_map, size = map_labels_to_imgs(img_dir, label_list, test, upper)

    if vocabulary is not None:
        words = get_word_list(vocabulary, upper, test)
        warn = True
    else:
        words = get_word_list(label_list, upper)
        warn = False
    phocs = compute_phocs(label_img_map.keys(), phoc_unigram_levels,
                          words, warn)

    lmdb_creator = setup_lmdb(output_dir, dataset_name, test)

    match_computed_phocs(img_dir, label_img_map, phocs, lmdb_creator,
                         size, n_train_images, augment)

    lmdb_creator.finish_creation()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create LMDB for training '
                                     'with given PHOC descriptors.')
    parser.add_argument('img_dir', help='Path to folder with training images.')
    parser.add_argument('label_list', help='File with label ids and '
                        'transcriptions')
    parser.add_argument('output_dir', help='Folder in which the LMDB '
                        'folders should be stored')
    parser.add_argument('dataset_name', help='Name of the dataset')
    parser.add_argument('--phoc_unigram_levels', '-pul', action='store',
                        type=lambda x: [int(elem) for elem in x.split(',')],
                        default='2,3,4,5', help='Comma seperated list of PHOC '
                        'unigram levels to be used when computing PHOCs. '
                        'Default: 2,3,4,5')
    parser.add_argument('--vocabulary', help='File containing vocabulary to '
                        'be used', default=None)
    parser.add_argument('--n_train_images', help='Total number of training '
                        'images to be produced.', default=None, type=int)
    parser.add_argument('--test', action='store_true', help='Create test LMDB')
    parser.add_argument('--upper', action='store_true', help='Convert words '
                        'in given .csv file to upper case.')
    parser.add_argument('--augment', choices=[TRANSFORM, RESIZE],
                        default=None, help='Indicate augmentation strategy '
                        'to be used.')

    args = vars(parser.parse_args())
    main(**args)
