# Import necessary libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# to get fonts suitable for talks/presentations
sns.set_context('talk')

# Filepaths
data_folder_path="data"


###
### CREATING THE DATASET
###

def ask_for_train_size(max_length=10000):
    train_size = None
    print("Number of training samples to be used:")
    print("Please choose a number between 100 and {max}".format(max=max_length))
    while train_size is None:
        input_value = input(" ")
        try:
            # try and convert the string input to a number
            train_size = int(input_value)
            if not 100<=train_size<=max_length:
                train_size = None
                print("Please choose a number between 100 and {max}".format(max=max_length))
        except ValueError:
            print("Please enter a number:")
    print("")
    return train_size
     
def ask_for_train_contamination():
    train_contam = None
    print("Portion of training samples to be attacks:")
    print("Please choose a number between 0 and 0.5 .")
    while train_contam is None:
        input_value = input(" ")
        try:
            # try and convert the string input to a float number
            train_contam = float(input_value)
            if not 0<=train_contam<=0.5:
                train_contam = None
                print("Please choose a number between 0 and 0.5 .")
        except ValueError:
            print("Please enter a number:")
    print("")
    return train_contam


def generate_training_data(data_full, labels_full, 
                       training_samples=10000, 
                       training_contamination=0.25):
    X=data_full
    y=labels_full
    training_samples=np.minimum(training_samples, y.shape[0])

    # Calculate subset sizes
    attack_filter=y.attack==1
    attack_filter.sum()
    n_attack = np.ceil(training_samples * training_contamination).astype('int')
    if n_attack>attack_filter.sum():
        # reduce number of attacks and fill with normal
        print("warning_msg")
        n_attack = attack_filter.sum()
    n_normal = training_samples - n_attack

    # Create subframes
    X_normal=X.loc[~attack_filter, :]
    y_normal=y.loc[~attack_filter, :]
    X_attack=X.loc[attack_filter, :]
    y_attack=y.loc[attack_filter, :]

    # Sample normal and attacks
    if n_attack>0:
        _, X_n, _, y_n = train_test_split(X_normal, y_normal, 
                                          test_size=n_normal, random_state=7)
        _, X_a, _, y_a = train_test_split(X_attack, y_attack, 
                                          test_size=n_attack, random_state=7)
        data_train = pd.concat([X_n, X_a])
        labels_train = pd.concat([y_n, y_a])
    else:
        _, data_train, _, labels_train = train_test_split(X_normal, y_normal, 
                                          test_size=n_normal, random_state=42)
    
    return data_train, labels_train


def label_counter(labels):
    label_counter = labels.attack_detail.value_counts().to_frame().rename(columns={"attack_detail": "frequency"})
    label_counter.index.set_names(['class'], inplace=True)
    return label_counter

def create_dataset():
    ### data_folder_path="data"
    # import data
    data_full=pd.read_csv(os.path.join(data_folder_path, "data_AMLD.csv"))
    labels_full=pd.read_csv(os.path.join(data_folder_path, "labels_AMLD.csv"))

    # get parameters from user
    training_samples = ask_for_train_size(max_length=data_full.shape[0])
    training_contamination = ask_for_train_contamination()
    # generate training data
    data_train, labels_train = generate_training_data(data_full, labels_full, 
                                                      training_samples=training_samples, 
                                                      training_contamination=training_contamination)
    
    # import test datasets
    #data_10=pd.read_csv(os.path.join(data_folder_path, "data_test_10K.csv"))
    #labels_10=pd.read_csv(os.path.join(data_folder_path, "labels_test_10K.csv"))
    data_1=pd.read_csv(os.path.join(data_folder_path, "data_test_1K.csv"))
    labels_1=pd.read_csv(os.path.join(data_folder_path, "labels_test_1K.csv"))
    # bundle datasets
    dataset={"data_train": data_train.astype(np.float64),
             "labels_train": labels_train,
             #"data_test10": data_10,
             #"labels_test10": labels_10,
             "data_test1": data_1.astype(np.float64),
             "labels_test1": labels_1,
             # default choice with_PCA=False, so test set is default test set
             "data_test1_PCA": data_1}
    
    print("Your dataset has been created.")
    return dataset


###
### VISUALISING THE DATA
###


def downsample_for_plot(data, labels, size):
    _, X, _, labels_plot = train_test_split(data, labels, test_size = size, 
                            random_state=42, stratify=labels.attack_detail)
    return X, labels_plot

# Build visualiser of data sets using TSNE
def TSNE_visualiser(X, labels_plot, colour_by_level=2, marker_list=[], perplexity=25):
    tSNE = TSNE(n_components=2, learning_rate=300,
            perplexity=perplexity, early_exaggeration=12,
            init='random', random_state=42)    
    
    X.reset_index(drop=True, inplace=True)
    labels_plot.reset_index(drop=True, inplace=True)
    
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)

    pca = PCA(n_components=None)
    X_pca=pca.fit_transform(X_scaled)
    
    X_tSNE = tSNE.fit_transform(X_pca)
    X_tSNE = pd.DataFrame(data=X_tSNE)
    
    colouring = labels_plot.iloc[:, colour_by_level]
    
    color_dict = dict({'normal':'lightblue',
                       'neptune':'green',
                       'smurf':'yellowgreen',
                       'teardrop': 'chartreuse',
                       'back': 'limegreen',
                       'pod': 'seagreen',
                       'satan': 'red',
                       'ipsweep': 'firebrick',
                       'portsweep': 'tomato',
                       'nmap':'orange',
                       'warezclient': 'darkorchid',
                       'guess_passwd': 'indigo'})
    
    
    plt.figure(figsize=(12,12))
    sns.scatterplot(x=X_tSNE[0], y=X_tSNE[1], hue=colouring, 
                    s=25, palette=color_dict)
    plt.legend()
    plt.show()


def visualise_training_data(dataset, max_plot=1000):
    print("Plotting", max_plot, "samples of the training data")
    
    data = dataset["data_train"]
    labels = dataset["labels_train"]
    if data.shape[0] > max_plot:
        data, labels = downsample_for_plot(data,
                                           labels, 
                                           size=max_plot)  
    TSNE_visualiser(data, labels, colour_by_level=2, perplexity=25)

def explore_and_visualise_training_data(dataset, max_plot=1000):
    # display training data info
    print("Your training set has the following composition:")
    display(label_counter(dataset["labels_train"]))
    visualise_training_data(dataset, max_plot=max_plot)
    
###
### BUILDING THE MODEL
###


def build_anomaly_detector(dataset, 
                           expected_contamination=0.1,
                           with_PCA = False):

    # Load training set
    X_train = dataset["data_train"]
    y_train = dataset["labels_train"]
    
    # Define X_test, y_test
    X_test = dataset["data_test1"]
    y_test = dataset["labels_test1"]#["attack"]
    
    # PCA option
    if with_PCA:
        print("Using PCA to transform data")
        # Fit_transform training set
        scaler_pca=StandardScaler()
        X_train_scaled=scaler_pca.fit_transform(X_train)
        pca = PCA(n_components=None)
        X_train=pca.fit_transform(X_train_scaled)
        # Apply to test set
        X_test_scaled=scaler_pca.transform(X_test)
        dataset["data_test1_PCA"]=pca.transform(X_test_scaled)
    else:
        dataset["data_test1_PCA"]=dataset["data_test1"]
    
    # Define IF-model
    clf= IsolationForest(n_estimators=100, 
                     max_samples='auto', 
                     max_features=float(1.0), 
                     contamination=float(expected_contamination), 
                     random_state=42, 
                     behaviour="new") 
    # train model
    clf.fit(X_train)
    return clf, dataset

############################    

def make_predictions(dataset,clf):
    # Load training set
    X_train = dataset["data_train"]
    y_train = dataset["labels_train"]
    
    # Define X_test, y_test
    X_test = dataset["data_test1_PCA"]
    y_test = dataset["labels_test1"]#["attack"]
    
    # make predictions on test set
    y_pred_clf = clf.predict(X_test)
    # in y_pred_clf outlier = -1 and normal = 0
    y_pred=pd.Series((y_pred_clf<0)*1)
    # create data frame with labels and predictions
    y_eval_df = y_test.copy()
    y_eval_df['pred']= y_pred
    return y_eval_df
############################    

def evaluate_model(dataset,clf):
    # Load training set
    X_train = dataset["data_train"]
    y_train = dataset["labels_train"]
    
    # Define X_test, y_test
    X_test = dataset["data_test1_PCA"]
    y_test = dataset["labels_test1"]#["attack"]
    
    y_eval_df=make_predictions(dataset,clf)
    ############################    
    # draw decision boundary
    n_bins=100
    scoring = clf.decision_function(X_test) 
    plot_min=1.1*np.min(scoring)
    plot_max=1.1*np.max(scoring)
    bins = np.linspace(plot_min, plot_max, n_bins)
    # Create plot
    fig=plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)    
    plt.hist(scoring[y_test.attack == 0], bins, alpha=0.5, label='True normal')
    plt.hist(scoring[y_test.attack == 1], bins, alpha=0.5, label='True attack')
    # Add decision boundary and zone labels
    plt.axvline(x=0, ls='--', color='r', label="Decision boundary")
    plt.text(0.2, 0.5,'Predicted attacks', ha='center', va='center', 
             transform=ax.transAxes, style='italic')
    plt.text(0.85, 0.5,'Predicted normal', ha='center', va='center', 
             transform=ax.transAxes, style='italic')
    plt.title('Outlier scores for test samples (below 0 assigned as outlier)')
    plt.xlim(left=plot_min, right=plot_max)
    plt.ylim(bottom=0, top=180)
    plt.legend()
    plt.show()
    ############################    
    # extract ground truth and predictions
    y_true =y_eval_df.attack
    y_pred=y_eval_df.pred
    cm = confusion_matrix(y_true, y_pred)
    # create and plot confusion_matrix
    plot_labels = ['Normal', 'Attack']
    plt.figure(figsize = (6, 5))
    heatmap=sns.heatmap(cm, annot=True, fmt = 'd')  # fmt = 'd' suppresses scientific notation
    heatmap.set(xlabel="Predicted", ylabel="True", 
                xticklabels=plot_labels, yticklabels=plot_labels)
    plt.show()
    
    # Summary
    print("Summary report:")

    print("Attacks identified correctly (true positive):", cm[1,1], " of ", cm[1,1]+ cm[1,0], "(",
          (100*cm[1,1]/(cm[1,1]+ cm[1,0])).round(1), "%)")
    print("Attacks missed (false negative):", cm[1,0], " of ", cm[1,1]+ cm[1,0], "(", 
          (100*cm[1,0]/(cm[1,1]+ cm[1,0])).round(1), "%)")
    print("False alarms (false positive):", cm[0,1], " of", cm[0,0]+ cm[0,1], "(", 
          (100*cm[0,1]/(cm[0,0]+ cm[0,1])).round(1), "%)")
    print("Total number of misclassified samples:", cm[0,1]+cm[1,0])
    
############################
def detailed_evaluation(dataset, clf, colour_by_level=2, perplexity=25, max_plot=1000):
    # Load training set
    X_train = dataset["data_train"]
    y_train = dataset["labels_train"]
    
    # Define X_test, y_test
    X_test = dataset["data_test1_PCA"]
    y_test = dataset["labels_test1"]#["attack"]
    
    y_eval_df=make_predictions(dataset,clf)

    print("Detailed report:")
    
    # filter for all misclassified data points
    misclassified=y_eval_df[y_eval_df['attack']!=y_eval_df['pred']]
    missed=misclassified[misclassified['pred']==0]
    # build reporting data frame with originally present types, 
    # number of misclassified per type and the percentage
    df1=pd.DataFrame(y_eval_df['attack_detail'].value_counts()).reset_index()
    df2=pd.DataFrame(missed['attack_detail'].value_counts()).reset_index()
    df=df1.merge(df2, how='outer', on='index').rename({'index':'attack_detail', 
                                                       'attack_detail_x': 'present', 
                                                       'attack_detail_y': 'missed'}, axis=1)
    df=df.fillna(0)
    df.set_index('attack_detail', inplace=True)
    # Correct missed normals from 0 to false alarms
    false_alarms=confusion_matrix(y_eval_df.attack, y_eval_df.pred)[0,1]
    df.loc['normal', 'missed']=false_alarms
    df.missed=df.missed.astype(int)
    df['pct_missed']=round(100*df.missed/df.present, 1)
    display(df)    
    ############################
    #  Plot tSNE plot with misclassified highlighted
    print("Plotting", max_plot, "samples of the test data")
    
    data = dataset["data_test1"].copy()
    labels = dataset["labels_test1"].copy()
    if data.shape[0] > max_plot:
        data, labels = downsample_for_plot(data,
                                           labels, 
                                           size=1000)

 
    labels_plot=labels
    X=data
    tSNE = TSNE(n_components=2, learning_rate=300,
            perplexity=perplexity, early_exaggeration=12,
            init='random', random_state=42)    
    
    X.reset_index(drop=True, inplace=True)
    labels_plot.reset_index(drop=True, inplace=True)
    
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)

    pca = PCA(n_components=None)
    X_pca=pca.fit_transform(X_scaled)
    
    X_tSNE = tSNE.fit_transform(X_pca)
    X_tSNE = pd.DataFrame(data=X_tSNE)
    
    wrong_predictions=y_eval_df['attack']!=y_eval_df['pred']
    X_tSNE_wrong=X_tSNE.loc[wrong_predictions, :]
    
    colouring = labels_plot.iloc[:, colour_by_level]
    
    color_dict = dict({'normal':'lightblue',
                       'neptune':'green',
                       'smurf':'yellowgreen',
                       'teardrop': 'chartreuse',
                       'back': 'limegreen',
                       'pod': 'seagreen',
                       'satan': 'red',
                       'ipsweep': 'firebrick',
                       'portsweep': 'tomato',
                       'nmap':'orange',
                       'warezclient': 'darkorchid',
                       'guess_passwd': 'indigo'})
    color_dict_errors = dict({'normal':'tab:blue',
                       'neptune':'green',
                       'smurf':'yellowgreen',
                       'teardrop': 'chartreuse',
                       'back': 'limegreen',
                       'pod': 'seagreen',
                       'satan': 'red',
                       'ipsweep': 'firebrick',
                       'portsweep': 'tomato',
                       'nmap':'orange',
                       'warezclient': 'darkorchid',
                       'guess_passwd': 'indigo'})

    
    plt.figure(figsize=(12,12))
    sns.scatterplot(x=X_tSNE[0], y=X_tSNE[1], hue=colouring, s=25, palette=color_dict)
    sns.scatterplot(x=X_tSNE_wrong[0], y=X_tSNE_wrong[1], hue=colouring, marker="^",
                    s=70, palette=color_dict_errors, legend=False)
    plt.legend()
    plt.show()

    
