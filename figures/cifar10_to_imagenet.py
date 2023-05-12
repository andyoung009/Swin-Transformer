import os
import shutil
import pickle
from PIL import Image
from torchvision.datasets import CIFAR10
import numpy as np

# 定义CIFAR-10数据集的路径
cifar10_dir = "D:\\Work\\datasets\\cifar10"
IMAGE_SIZE = 32
# 定义ImageNet格式的数据集路径
imagenet_dir = "D:\\Work\\datasets\\cifar10_2_imagenet"


def convert_cifar10_to_imagenet(root):
    data_dir = os.path.join(root, 'cifar-10-batches-py')
    train_dir = os.path.join(root, 'cifar10_imagenet/train')
    val_dir = os.path.join(root, 'cifar10_imagenet/val')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # convert training set
    for i in range(1, 6):
        data_file = os.path.join(data_dir, 'data_batch_{}'.format(i))
        with open(data_file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            images = data[b'data']
            labels = data[b'labels']
            num_images = images.shape[0]

            for j in range(num_images):
                image = images[j]
                label = labels[j]
                image_dir = os.path.join(train_dir, str(label))
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                image_file = os.path.join(image_dir, '{}_{}.jpg'.format(label, j))
                # image = image.resize((224,224))
                save_image(image, image_file)

    # convert validation set
    data_file = os.path.join(data_dir, 'test_batch')
    with open(data_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        images = data[b'data']
        labels = data[b'labels']
        num_images = images.shape[0]

        for j in range(num_images):
            image = images[j]
            label = labels[j]
            image_dir = os.path.join(val_dir, str(label))
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            image_file = os.path.join(image_dir, '{}_{}.jpg'.format(label, j))
            # image = image.resize((224,224))
            save_image(image, image_file)

def save_image(image, filename):
    # # convert image data to RGB format
    # image = image.reshape((3, 32, 32)).transpose((1, 2, 0))
    # # save image file
    # image = Image.fromarray(image)

    # change the resolution to (224,224)
    image = Image.fromarray(np.transpose(np.reshape(image, (3, IMAGE_SIZE, IMAGE_SIZE)), (1, 2, 0)))            
    # 调整图片大小
    resized_image = image.resize((224, 224))


    resized_image.save(filename)

# Example usage:
root = 'D:\\Work\\datasets\\cifar10'
convert_cifar10_to_imagenet(root)
