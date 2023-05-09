# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)
# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor, ConvertImageDtype, Compose
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2

#from customdataset import CustomImageDataset


# set the device we will be using to test the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# testData = CustomImageDataset(
#     annotations_file='friburgo_train_annotations_file.csv',
#     img_dir='',
# 	transform=Compose([
# 						ConvertImageDtype(torch.float32)
# 						])
# )

#idxs = np.random.choice(range(0, len(testData)), size=(10,))
#testData = Subset(testData, idxs)
# initialize the test data loader
#testDataLoader = DataLoader(testData, batch_size=2)
#print(testDataLoader)
# load the model and set it to evaluation mode
#model = torch.load(args["model"]).to(device)
print('LOAD MODEL FROM DISK')
model=torch.load('dummy.pth').to(device)
print('EVALUATE MODEL')
model.eval()

print('MAKING PREDICTION')
# Load the image and preprocess it
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix

# img_path_csv=pd.read_csv('friburgo_test_dataset_filenames.csv', header=0)
# #print(img_path_csv.iloc[0])
# #for path in img_path_csv.tolist():
# img_path = 'Friburgo/Friburgo_Test/t1152902911.896518_x-5.460308_y-4.897937_a-0.651417.jpeg'
# #img_path=path
# print(img_path)
# image = Image.open(img_path)
# transform = transforms.Compose([
#     #transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# input_tensor = transform(image).unsqueeze(0) #bueno pues esto ya funciona, se lo traga la red, ahora es cuestion de automatizar para coger varias fotos o algo
# # Make a prediction on the preprocessed image
# with torch.no_grad():
#     output = model(input_tensor)

# # Get the predicted class label
# _, predicted = torch.max(output, 1)
# class_index = predicted.item()

# # Print the predicted class label
# print('PREDICTION IS: ')
# print(class_index)

# Load your test dataset
test_csv = 'friburgo_test_annotations_file.csv'
test_df = pd.read_csv(test_csv)

# Create empty lists to store the predicted labels and ground truth labels
predicted_labels = []
true_labels = []

# Loop through the test dataset and make predictions
for i in range(len(test_df)):
    img_path = test_df.iloc[i, 0]
    image = Image.open(img_path).convert('L') #convert to grayscale
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output.data, 1)
    predicted_label = predicted.item()
    true_label = test_df.iloc[i, 1]
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Print the matrix with diagonal highlighted
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        if i == j:
            # Use a special character to highlight the diagonal element
            print('*{value}*'.format(value=conf_matrix[i][j]), end=' ')
        else:
            print(conf_matrix[i][j], end=' ')
    print()

# Now to visualize the confusion matrix as probabilities
# Calculate the column sums
column_sums = np.sum(conf_matrix, axis=0)

# Divide each element in the matrix by its corresponding column sum
normalized_matrix = conf_matrix / column_sums

# Print the normalized matrix
# Print the matrix with diagonal highlighted
for i in range(len(normalized_matrix)):
    for j in range(len(normalized_matrix)):
        if i == j:
            # Use a special character to highlight the diagonal element
            print('*{value}*'.format(value=normalized_matrix[i][j]), end=' ')
        else:
            print(normalized_matrix[i][j], end=' ')
    print()

import matplotlib.pyplot as plt

# Create a colormap that goes from red to green
cmap = plt.cm.get_cmap('RdYlGn')

# Plot the heatmap
plt.imshow(normalized_matrix, cmap=cmap)

# Add a colorbar
plt.colorbar()

# Show the plot
plt.show()

# de este codigo de abajo podría sacar una visualizacion chula, pero no es muy util
# # switch off autograd
# with torch.no_grad():
# 	# loop over the test set
# 	for (image, label) in testDataLoader:
# 		#print(image)
# 		print(label)
# 		# grab the original image and ground truth label
# 		origImage = image.numpy().squeeze(axis=(0, 1))
# 		gtLabel = testData.dataset.classes[label.numpy()[0]]
# 		# send the input to the device and make predictions on it
# 		image = image.to(device)
# 		pred = model(image)
# 		# find the class label index with the largest corresponding
# 		# probability
# 		idx = pred.argmax(axis=1).cpu().numpy()[0]
# 		predLabel = testData.dataset.classes[idx]
		
# 		# convert the image from grayscale to RGB (so we can draw on
# 		# it) and resize it (so we can more easily see it on our
# 		# screen)
# 		origImage = np.dstack([origImage] * 3)
# 		origImage = imutils.resize(origImage, width=128)
# 		# draw the predicted class label on it
# 		color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
# 		cv2.putText(origImage, gtLabel, (2, 25),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
# 		# display the result in terminal and show the input image
# 		print("[INFO] ground truth label: {}, predicted label: {}".format(
# 			gtLabel, predLabel))
# 		cv2.imshow("image", origImage)
# 		cv2.waitKey(0)