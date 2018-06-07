import tensorflow as tf 
import numpy as np 
from six.moves import cPickle as pickle 
from six.moves import range

#Import Data
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_label']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save
	print('Training Set', train_dataset.shape, train_labels.shape)
	print('Validation Set', valid_dataset.shape, valid_labels.shape)
	print('Test Set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

#Reformat the data for the neural network
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
	#Map 0 to [1.0,0,0....], 1 to [0.0, 1.0, 0.0....]
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels2 = reformat(train_dataset, train_labels)
valid_dataset,valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels2  = reformat( test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels2.shape)
print('Validation Set', valid_dataset.shape, valid_labels.shape)
print('Test Set', test_dataset.shape, test_labels.shape )
print(train_labels)

train_subset = 10000


#print(train_labels.shape)

#print(train_dataset[1])
#print(train_dataset[1].shape)

#input values for Classifier 

def input_fn(train_dataset, train_labels):
	features  = {}
	for a in range(0,train_dataset.shape[1]):
		arr = train_dataset[:,a]
		#print(a)
		#print(arr.shape)
		features[str(a)] = arr

	#print(features.keys())
	#print(features.get('500').shape)
	labels = train_labels
	return features, labels

input_features, input_labels = input_fn(train_dataset[:train_subset,:],train_labels[:train_subset])
test_output_features, test_output_labels = input_fn(test_dataset,test_labels)
	
feature_columns = []
for key in input_features.keys():
	feature_columns.append(tf.feature_column.numeric_column(key=key))

def train_input_fn(features,labels,batch_size):
	dataset = tf.data.Dataset.from_tensor_slices((features,labels))
	return dataset.batch(batch_size)

classifier = tf.estimator.DNNClassifier(feature_columns = feature_columns, hidden_units = [1024], n_classes = 10 )
print("Classifier initialized")
classifier.train(input_fn = lambda: train_input_fn(input_features,input_labels,500), steps = 8000)
print("Classifier trained")
eval_result = classifier.evaluate(input_fn=lambda:train_input_fn(test_output_features,test_output_labels,500))
print('Test Set accuracy:{accuracy:0.3f}'.format(**eval_result))



