#wine quality dataset 
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pdb


OUTPUT_PATH= "WineQualityDataset.csv"

#headers
HEADERS=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]


def read_data(path):
    """
    Read the data into pandas dataframe
    :param path:
    :return:
    """
    data = pd.read_csv(path)

    return data
def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param dataset: Pandas dataframe
    :return: None, print the basic statistics of the dataset
    """
    print dataset.describe()

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def kfold(dataset, target):
    k = KFold(n_splits = 10)
    accuracy_training= 0
    accuracy_predictor = 0
    fold_score=0
    for x,y in k.split(dataset):
	train_x, test_x =dataset.iloc[x], dataset.iloc[y]
	train_y, test_y = target[x], target[y]
	print "Train_x Shape :: ", train_x.shape
    	print "Train_y Shape :: ", train_y.shape
   	print "Test_x Shape :: ", test_x.shape
    	print "Test_y Shape :: ", test_y.shape
    	# Create random forest classifier instance
    	trained_model = random_forest_classifier(train_x, train_y)
    	print "Trained model :: ", trained_model
    	predictions = trained_model.predict(test_x)
    	for i in xrange(0, 5):
		print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])
	fold_score = accuracy_score(test_y, predictions)
	accuracy_training+=accuracy_score(train_y, trained_model.predict(train_x))
	accuracy_predictor+= fold_score
	print "Fold Score" , fold_score
    	print " Confusion matrix ", confusion_matrix(test_y, predictions)
	random_forest_classifier(train_x, train_y)

    trainAccuracy = accuracy_training/10
    testAccuracy = accuracy_predictor/10
    print "Train Accuracy " , trainAccuracy
    print  "TestAccuracy" , testAccuracy




def main():
    """
    Main function
    :return:
    """
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH)
    # Get basic statistics of the loaded dataset
    dataset_statistics(dataset)
    target= dataset["quality"]
    dataset.drop('quality', axis=1)
    kfold(dataset, target)
    #train_x, test_x, train_y, test_y = split_dataset(dataset, 0.9,HEADERS[1:-1], HEADERS[-1])
    # Train and Test dataset size details
    #print "Train_x Shape :: ", train_x.shape
    #print "Train_y Shape :: ", train_y.shape
    #print "Test_x Shape :: ", test_x.shape
    # print "Test_y Shape :: ", test_y.shape
    # Create random forest classifier instance
    # trained_model = random_forest_classifier(train_x, train_y)
    #print "Trained model :: ", trained_model
    #predictions = trained_model.predict(test_x)
   # for i in xrange(0, 5):
      #	print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])
    #print "Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))
    #print "Test Accuracy  :: ", accuracy_score(test_y, predictions)
   # print " Confusion matrix ", confusion_matrix(test_y, predictions)

if __name__ == "__main__":
    main()

