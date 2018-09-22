import glob
import numpy as np
import cv2


def shuffle_samples(images, labels):
    index = list(range(len(labels)))
    np.random.shuffle(index)
    return images[index], labels[index]

def get_rotated_image(img, angle):
    (rows, cols, ch) = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def get_multiple_rotated_images(image, number_of_rotation):
    rotated_image_list = []
    for i in range(number_of_rotation):
        angle = np.random.randint(0, 360)
        img = get_rotated_image(image, angle)
        rotated_image_list.append(img)
    return rotated_image_list


def process_image(image_path, image_size_tuple):
    img = cv2.imread(image_path)
    img = (img / 255.).astype(np.float32)
    if img is not None:
        img = cv2.resize(img, image_size_tuple)
        return img


def load_image_with_label(images_path_list, label, image_size_tuple, n_rotated):
    x = []
    y = []
    for path in images_path_list:
        img = cv2.imread(path)
        img = (img / 255.).astype(np.float32)
        if img is not None:
            img = cv2.resize(img, image_size_tuple)
            x.append(img)
            y.append(label)
            if n_rotated > 0:
                rotated_images = get_multiple_rotated_images(img, n_rotated)
                x.extend(rotated_images)
                y.extend([label] * len(rotated_images))
    return x, y


def load_images(hotdog_images_path, not_hotdog_images_path, img_size, n_rotated):
    # get all the hotdog and not hotdog images paths
    # load hotdog images and labels. eg: [image1, image2, image3,1]      [1, 1, 1]
    hotdog_images, hotdog_labels = load_image_with_label(hotdog_images_path, 1, img_size, n_rotated)
    # load not hotdog images and labels. eg: [image1, image2, image3,1]    [0, 0, 0]
    not_hotdog_images, not_hotdog_labels = load_image_with_label(not_hotdog_images_path, 0, img_size, n_rotated)
    all_images = np.concatenate((hotdog_images, not_hotdog_images), axis=0)
    all_labels = np.concatenate((hotdog_labels, not_hotdog_labels), axis=0)
    return all_images, all_labels
