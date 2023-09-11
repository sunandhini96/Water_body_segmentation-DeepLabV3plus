# importing all packages

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import  accuracy_score, f1_score, jaccard_score, precision_score, recall_score, confusion_matrix , classification_report
from deeplabv3Ex import Deeplabv3


from metrics import dice_coef, iou , cal_metrics_pre_re


#to load test data 
def load_data(path):
    X = sorted(glob(os.path.join(path, "RGB", "*png")))
    Y = sorted(glob(os.path.join(path, "mask", "*png")))
    return X, Y



""" Global parameters """
H = 100
W = 100

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(y_pred, save_image_path):


    y_pred = np.expand_dims(y_pred, axis=-1)    ## (100, 100, 1)

    y_pred = y_pred * 255

    #pred_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, y_pred)



if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("split_results/all_files")

    """ Loading model """
    checkpoint_dir = "training_50/"
    os.listdir(checkpoint_dir)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print("latest checkpoint", latest)
    model = Deeplabv3((H, W, 3))
    model.load_weights(latest)

    """ Load the dataset """
  
    dataset_path = "/home/vv_sajithvariyar/sunandini_dir/January_sunandini/Test_data/"
    #Test_path = os.path.join(dataset_path, "test_dataset")
    test_x, test_y = load_data(dataset_path)
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Evaluation and Prediction """
    SCORE = []
    y_true_all=np.empty(0,dtype=np.int32)
    y_pred_all=np.empty(0,dtype=np.int32)
    
    
    
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".png")[0]
        name=name.split("\\")[-1]

        """ scaling the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.0
        x = np.expand_dims(x, axis=0)
        #x=x.astype(np.float32)

        """ scaling the mask """
        #thresh=128
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        #mask=cv2.threshold(mask,thresh,255,cv2.THRESH_BINARY)[1]
        mask=mask/255.0
        
        """ Prediction """
        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
#        print(y_pred)
#        print(mask)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)
        mask=mask.astype(np.int32)
        """ Saving the prediction """
        save_image_path = f"split_results/all_files/{name}.png"
        save_results(y_pred, save_image_path)

        """ Flatten the array """
        mask = mask.flatten()
 #       print(mask)
        y_pred = y_pred.flatten()
        print(y_pred)
        
        y_true_all=np.concatenate((y_true_all,mask))
        y_pred_all=np.concatenate((y_pred_all,y_pred))
        
        """ Calculating the metrics values """

        acc_value = accuracy_score(mask, y_pred)
        f1_value = f1_score(mask, y_pred, labels=[0, 1], average=None)
        jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average=None)
        recall_value = recall_score(mask, y_pred, labels=[0, 1], average=None)
        precision_value = precision_score(mask, y_pred, labels=[0, 1], average=None)
     
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])
        #print("SCORE", SCORE)
    
    confusion_matrix_score = confusion_matrix(y_true_all, y_pred_all , labels = [0,1])
    classification_report_score = classification_report(y_true_all, y_pred_all , labels=[0,1])
    print("confusion matrix",confusion_matrix_score)
    print("classification report",classification_report_score)  
    
    num_classes = confusion_matrix_score.shape[0]
    TP = np.diag(confusion_matrix_score)
    FP = np.sum(confusion_matrix_score, axis=0) - TP
    FN = np.sum(confusion_matrix_score, axis=1) - TP
    jaccard_scores = TP / (TP + FP + FN)
    mean_jaccard_score = np.mean(jaccard_scores)

# print the Jaccard score for each class and the mean Jaccard score
    print('Jaccard scores:', jaccard_scores)
    print('Mean Jaccard score:', mean_jaccard_score)
    """ Metrics values """
    #score = [s[1:]for s in SCORE]
    print("jaccard score",jaccard_score(y_true_all,y_pred_all,labels=[0,1],average=None))
    print("f1_score",f1_score(y_true_all,y_pred_all,labels=[0,1],average=None))
    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score_all_test_data.csv")


# In[ ]:




