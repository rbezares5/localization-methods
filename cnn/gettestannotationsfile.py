import pandas as pd
from scipy.spatial import distance
import numpy as np

def get_xy(name):
    lista=name.split('_',-1)
    for word in lista:
        if word[0]=='x':
            x=word[1:]
            x=float(x)
        if word[0]=='y':
            y=word[1:]
            y=float(y)
    return x,y


def create_annotations(train_file, test_file, codename):
    """
    Create annotations file combining 2 csv files
    """
    # df = pd.concat(map(pd.read_csv, [imgs_file, info_file]), ignore_index=True, axis=1)
    # print(df)

    # pd.DataFrame(df).to_csv('{name}_annotations_file.csv'.format(name=codename), index=None, header=None)

    # Read the csv files using pandas
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Extract the the data from dataframes as separate lists
    train_imgs = train_df.iloc[:, 0].tolist() # extract first column and convert to list of str
    train_labels = train_df.iloc[:, 1].astype(int).tolist() # extract second column and convert to list of int
    test_imgs = test_df.iloc[:, 0].tolist()

    # Make a list of all train coordinates
    train_coords=[]
    for file in train_imgs:
        x,y=get_xy(file)
        train_coords.append([x,y])

    test_labels=[]

    # Compare test coords with train coords
    for i in range(len(test_imgs)):
        x_test,y_test = get_xy(test_imgs[i])

        # First, make a list of all metric distances
        dist_list=[]
        for k in range(len(train_coords)):
            [x_train,y_train]=train_coords[k][:]
            dist=distance.euclidean([x_test,y_test],[x_train,y_train])
            dist_list.append(dist)

        # Then, get smallest distance index
        idx = np.argmin(dist_list)

        # Assign corresponding label
        label = train_labels[idx]
        test_labels.append(label)


    # Save and export the dataframe
    data = {'file': test_imgs, 'label': test_labels}
    pd.DataFrame(data).to_csv('{name}_annotations_file.csv'.format(name=codename), index=None, header=None)


def main():
    create_annotations(train_file='friburgo_train_annotations_file.csv',
                       test_file='friburgo_test_dataset_filenames.csv',
                       codename='friburgo_test')




    




if __name__ == "__main__":
    main()