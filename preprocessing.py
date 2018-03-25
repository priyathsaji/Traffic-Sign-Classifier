import os
import matplotlib.pyplot as plt 
import skimage.data as data
from skimage import transform
from skimage.color import rgb2grey
import random
import numpy as np


def load_data(path):
	directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
	labels = []
	images = []

	for d in directories:
		label_directory = os.path.join(path,d)
		file_names = [os.path.join(label_directory,f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
		for f in file_names:
			images.append(data.imread(f))
			vector = [0 for i in range(62)]
			vector[int(d)] = 1
			labels.append(vector)

	c = list(zip(images,labels))
	random.shuffle(c)
	images,labels = zip(*c)
	image28 = [transform.resize(image,(28,28)) for image in images]
	image28 = np.array(image28)
	image28 = rgb2grey(image28)
	image28 = [np.reshape(image,[784]) for image in image28]
	image28 = np.array(image28)
	labels = np.array(labels)


	return image28,labels



