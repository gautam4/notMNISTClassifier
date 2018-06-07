import pickle
import numpy as np 

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 

#Logistic Classifier using scikit-learn to classify the letter images 



#Open file
file = open('noTMNIST.pickle', 'rb')
data = pickle.load(file)
train_dataset = data["train_dataset"]
train_label = data["train_label"]
valid_dataset = data["valid_dataset"]
valid_labels = data["valid_labels"]
test_dataset = data["test_dataset"]
test_labels = data["test_labels"]

#Print to test if correct datasets are loaded
print("train dataset")
print(train_dataset.shape)
print("Train labels")
print(train_label.shape)
print("Valid dataset")
print(valid_dataset.shape)
print("valid labels")
print(valid_labels.shape)
print("test dataset")
print(test_dataset.shape)
print("test labels")
print(test_labels.shape)

#Plot sample letter images to visualize the dataset
plt.figure(figsize = (20,5))
for index, (image, label) in enumerate(zip(train_dataset[0:5,:,:],train_label[0:5])):
	plt.subplot(1,5,index+1)
	plt.imshow(image, cmap = plt.cm.plasma)
	plt.title('Training: %i\n' % label, fontsize = 20)



plt.figure(figsize = (20,5))
for index, (image, label) in enumerate(zip(valid_dataset[0:5,:,:],valid_labels[0:5])):
	plt.subplot(1,5,index+1)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.title('Validation: %i\n' % label, fontsize = 20)

plt.figure(figsize = (20,6))
for index, (image, label) in enumerate(zip(test_dataset[0:5,:,:],test_labels[0:5])):
	plt.subplot(1,5,index+1)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.title('Testing: %i\n' % label, fontsize = 20)

plt.show()

#reshape input data for sklearn fit function
count, imX, imY = train_dataset.shape
Rshape_train_dataset = train_dataset.reshape((count, imX*imY))

count, imX, imY = test_dataset.shape
Rshape_test_dataset = test_dataset.reshape((count, imX*imY))


#Create 50 Sample Model 
logisticRegress50 = LogisticRegression(solver = 'lbfgs')
logisticRegress50.fit(Rshape_train_dataset[0:50,:],train_label[0:50])

score = logisticRegress50.score(Rshape_test_dataset,test_labels)
print("The accuracy of 50 sample classifier is ")
print(score)

#Create 100 Sample Model
logisticRegress100 = LogisticRegression(solver = 'lbfgs')
logisticRegress100.fit(Rshape_train_dataset[0:100,:],train_label[0:100])

score = logisticRegress100.score(Rshape_test_dataset,test_labels)
print("The accuracy of 100 sample classifier is ")
print(score)

#Create 1000 Sample Model
logisticRegress1000 = LogisticRegression(solver = 'lbfgs')
logisticRegress1000.fit(Rshape_train_dataset[0:1000,:],train_label[0:1000])

score = logisticRegress1000.score(Rshape_test_dataset,test_labels)
print("The accuracy of 1000 sample classifier is ")
print(score)

#Create 5000 Sample Model
logisticRegress5000 = LogisticRegression(solver = 'lbfgs')
logisticRegress5000.fit(Rshape_train_dataset[0:5000,:],train_label[0:5000])

score = logisticRegress5000.score(Rshape_test_dataset,test_labels)
print("The accuracy of 5000 sample classifier is ")
print(score)


#Create an all sample trained model
logisticRegressAll = LogisticRegression(solver = 'lbfgs')
logisticRegressAll.fit(Rshape_train_dataset,train_label)

score = logisticRegressAll.score(Rshape_test_dataset,test_labels)
print("The accuracy of the all sample classifier is ")
print(score)





