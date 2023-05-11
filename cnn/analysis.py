# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay



def display_matrix(conf_matrix, style):
    if style=='text':
        # Print the conf_matrix with diagonal highlighted
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix)):
                if i == j:
                    # Use a special character to highlight the diagonal element
                    print('*{value}*'.format(value=conf_matrix[i][j]), end=' ')
                else:
                    print(conf_matrix[i][j], end=' ')
            print()

    if style=='heatmap':
        # Now to visualize the confusion conf_matrix as probabilities
        # Calculate the column sums
        column_sums = np.sum(conf_matrix, axis=0)

        # Divide each element in the conf_matrix by its corresponding column sum
        normalized_matrix = conf_matrix / column_sums

        # Print the normalized conf_matrix
        # Print the conf_matrix with diagonal highlighted
        # for i in range(len(normalized_matrix)):
        #     for j in range(len(normalized_matrix)):
        #         if i == j:
        #             # Use a special character to highlight the diagonal element
        #             print('*{value}*'.format(value=normalized_matrix[i][j]), end=' ')
        #         else:
        #             print(normalized_matrix[i][j], end=' ')
        #     print()


        # Create a colormap that goes from red to green
        cmap = plt.colormaps.get_cmap('RdYlGn')

        # Plot the heatmap
        plt.imshow(normalized_matrix, cmap=cmap)

        # Add a colorbar
        plt.colorbar()

        # Show the plot
        plt.show()

    if style=='sklearn':
        # Alternative display
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot()
        plt.show()

def display_metrics(conf_matrix, weights=None):
    # Now we are going to compute analysis metrics
    TP = []
    FP = []
    FN = []
    TN = []
    accuracy = []
    precision = []
    recall = []
    specificity = []
    F1_score = []
    num_classes=len(conf_matrix)
    total = np.sum(conf_matrix)

    # Compute analysis metrics for each class
    for i in range(num_classes):
        # Extract the diagonal elements
        tp = conf_matrix[i][i]
        # Calculate the sum of all other elements in the same column
        fp = sum(conf_matrix[j][i] for j in range(num_classes) if j != i)
        # Calculate the sum of all other elements in the same row
        fn = sum(conf_matrix[i][j] for j in range(num_classes) if j != i)
        # Calculate the sum of all other elements outside the i-th row and column
        tn = total - tp - fp - fn
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
        TN.append(tn)
        accuracy.append((tp+tn)/total)
        precision.append(tp/(tp+fp))
        recall.append(tp/(tp+fn))
        specificity.append(tn/(tn+fp))
        F1_score.append((2*precision[i]*recall[i])/(precision[i] + recall[i]))


    # Compute and print global analysis metrics
    F1_micro = np.sum(TP)/(np.sum(TP)+np.sum(FP))
    print('Micro Accuracy: {value}'.format(value=F1_micro))
    F1_micro = np.sum(TP)/(np.sum(TP)+np.sum(FN))
    print('Micro Precision: {value}'.format(value=F1_micro))
    print('Micro Recall: {value}'.format(value=F1_micro))
    print('Micro F1-score: {value}'.format(value=F1_micro))
    print('')

    acc_macro = np.average(accuracy)
    print('Macro Accuracy: {value}'.format(value=acc_macro))
    prec_macro = np.average(precision)
    print('Macro Precission: {value}'.format(value=prec_macro))
    rec_macro = np.average(recall)
    print('Macro Recall: {value}'.format(value=rec_macro))
    F1_macro = np.average(F1_score)
    print('Macro F1-score: {value}'.format(value=F1_macro))
    print('')

    if weights.all()!=None:
        counts=weights
        acc_weighted = np.average(accuracy, weights=counts)
        print('Weighted Accuracy: {value}'.format(value=acc_weighted))
        prec_weighted = np.average(precision, weights=counts)
        print('Weighted Precission: {value}'.format(value=prec_weighted))
        rec_weighted = np.average(recall, weights=counts)
        print('Weighted Recall: {value}'.format(value=rec_weighted))
        F1_weighted = np.average(F1_score, weights=counts)
        print('Weighted F1-score: {value}'.format(value=F1_weighted))

def main():
    # Read the conf_matrix from the CSV file and convert the DataFrame to a numpy array
    conf_matrix = pd.read_csv('lenet_conf_matrix.csv', header=None).to_numpy()
    #print(conf_matrix)

    # Get the frequency of each label
    # Read the CSV file
    df = pd.read_csv('friburgo_test_annotations_file.csv')
    # Get the second column as a numpy array
    column_values = df.iloc[:, 1].values
    # Count the frequency of each number using numpy
    labels, counts = np.unique(column_values, return_counts=True)
    # Create a dictionary to store the frequency of each number
    #labels_frequency = dict(zip(labels, counts))
    # Print the frequency of each number
    #print(labels_frequency)
    #print(counts)

    # Display results
    display_matrix(conf_matrix, style='heatmap')
    display_metrics(conf_matrix, counts)

if __name__ == "__main__":
    main()