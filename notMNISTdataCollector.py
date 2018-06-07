#import modules

from __future__ import print_function
import imageio
import matplotlib.pyplot as plt 
import numpy as np 
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression 
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle 
import random
from random import randint

#Derived from Udacity's 1_notmnist.ipynb example.
#Downloads and extracts relevant dataset. 
#Creates a training, validation,and test dataset for later use by a classifier
#Stores the three datasets into a pickle file

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percept_reported = None
data_root = '.'

def download_progress_hook(count, blockSize,totalSize):
	global last_percept_reported
	percent = int(count*blockSize*100/totalSize)

	if last_percept_reported != percent:
		if percent % 5 == 0:
			sys.stdout.write("%s%%" % percent)
			sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()

		last_percept_reported = percent


def maybe_download(filename, expected_bytes, force=False):
	destination_fileName = os.path.join(data_root,filename)
	if force or not os.path.exists(destination_fileName):
		print('Attempting to downlaod:', filename)
		filename, _ = urlretrieve(url + filename, destination_fileName, reporthook = download_progress_hook)
		print('\nDownload Complete!')
	statinfo = os.stat(destination_fileName)
	if statinfo.st_size == expected_bytes:
		print('Found and Verified', destination_fileName)
	else:
		raise Exception('Failed to verify ' + destination_fileName + '. Can you get to it with a browser?')
	return destination_fileName

#Download the dataset
train_fileName = maybe_download('notMNIST_large.tar.gz', 247336696)
test_fileName = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes=10
np.random.seed(133)

def maybe_extract(fileName,force=False):
	root = os.path.splitext(os.path.splitext(fileName)[0])[0]
	if os.path.isdir(root) and not force:
		print('%s already present - Skipping extraction of %s.' % (root,fileName))
	else:
		print('Extracting data for %s. This may take a while. Please wait.' % root)
		tar = tarfile.open(fileName)
		sys.stdout.flush()
		tar.extractall(data_root)
		tar.close()
	data_folders = [
		os.path.join(root,d) for d in sorted(os.listdir(root))
		if os.path.isdir(os.path.join(root,d))]
	if len(data_folders) != num_classes:
		raise Exception('Expected %d folders, one per class. Found %d instead.' % (num_classes,len(data_folders)))
	print(data_folders)
	return data_folders

train_folders = maybe_extract(train_fileName)
test_folders = maybe_extract(test_fileName)

image_size = 28
pixel_depth = 255.0

def load_letter(folder, min_num_images):
	image_files = os.listdir(folder)
	dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)

	print(folder)
	num_images = 0
	for image in image_files:
		image_file=os.path.join(folder,image)
		try:
			image_data = (imageio.imread(image_file).astype(float)- pixel_depth/2)/pixel_depth
			if image_data.shape != (image_size, image_size):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			dataset[num_images, :, :] = image_data
			num_images = num_images +1
		except (IOError, ValueError) as e:
			print('Could not read:', image_file, ':', e, '- it\'s okay, skipping. ')

	dataset = dataset[0:num_images,:,:]
	if num_images < min_num_images:
		raise Exception('Many fewer images than expected:%d < %d' % (num_images, min_num_images))
	print('Full dataset tensor:', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('StDEV:', np.std(dataset))
	return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force = False):
	dataset_names = []
	for folder in data_folders:
		set_filename = folder + '.pickle'
		dataset_names.append(set_filename)
		if os.path.exists(set_filename) and not force:
			print('%s already present -Skipping pickling.' % set_filename)
		else:
			print('Pickling %s.' % set_filename)
			dataset = load_letter(folder, min_num_images_per_class)
			try:
				with open(set_filename, 'wb') as f:
					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print('Unable to save data to', set_filename, ':', e)
	return dataset_names

train_datasets = maybe_pickle(train_folders,45000)
test_datasets = maybe_pickle(test_folders, 1800)

def make_arrays(nb_rows, img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
	num_classes = len(pickle_files)
	valid_dataset2, valid_labels2 = make_arrays(valid_size, image_size)
	train_dataset2, train_labels2 = make_arrays(train_size,image_size)
	vsize_per_class = valid_size // num_classes
	tsize_per_class = train_size // num_classes

	start_v, start_t = 0,0
	end_v, end_t = vsize_per_class, tsize_per_class
	end_l = vsize_per_class+tsize_per_class
	for label, pickle_file in enumerate(pickle_files):
		try:
			with open(pickle_file, 'rb') as f:
				letter_set = pickle.load(f)
				np.random.shuffle(letter_set)

				if valid_dataset2 is not None:
					valid_letter = letter_set[:vsize_per_class,:,:]
					valid_dataset2[start_v:end_v,:,:] = valid_letter
					valid_labels2[start_v:end_v] = label
					start_v+= vsize_per_class
					end_v += vsize_per_class

				train_letter = letter_set[vsize_per_class:end_l,:,:]
				train_dataset2[start_t:end_t,:,:] = train_letter
				train_labels2[start_t:end_t] = label
				start_t += tsize_per_class
				end_t += tsize_per_class
		except Exception as e:
			print('Unable to process data from', pickle_file,':',e)
			raise
	return valid_dataset2, valid_labels2, train_dataset2, train_labels2

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset2,valid_labels2,train_dataset2,train_labels2 = merge_datasets(train_datasets,train_size, valid_size)
_,_, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training', train_dataset2.shape, train_labels2.shape)
print('Validation', valid_dataset2.shape, valid_labels2.shape)
print('Testing', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset,shuffled_labels

train_dataset2, train_labels2 = randomize(train_dataset2,train_labels2)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset2, valid_labels2 = randomize(valid_dataset2, valid_labels2)

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
	f = open(pickle_file, 'wb')
	save = {
		'train_dataset': train_dataset2,
		'train_label': train_labels2,
		'valid_dataset': valid_dataset2,
		'valid_labels': valid_labels2,
		'test_dataset': test_dataset,
		'test_labels': test_labels,
	}
	pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print('Unable to save data to', pickle_file, ':', e)
	raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

