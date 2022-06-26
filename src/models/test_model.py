# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../src/data/")

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from dataloader import *
from tqdm import tqdm
from config import *
from utils import *
from loss import *
import numpy as np
import logging
import os

def main():
    """ testing performance of model. 
    """
    trainGen, testGen = getDataLoader(batch_size = 1)
    model = tf.keras.models.load_model(os.path.join(WEIGHTS_DIR, 'ACLNet_Best.h5'), custom_objects={"diceCoef":diceCoef, "bceDiceLoss":bceDiceLoss})
    model.evaluate(testGen)
    original, prediction = [], []
    with tqdm(total = int(6468*TEST_SIZE)) as pbar:
        for data in testGen:
            image, mask = data
            seg = model.predict(image)
            original.append(mask[0].argmax(-1))
            prediction.append(seg[0].argmax(-1))
            pbar.update(1)

    original = np.array(original)
    prediction = np.array(prediction)

    precisions, recalls, f1scores, error_rates = [], [], [], []
    with tqdm(total = len(original)) as pbar:
        for orig, pred in zip(original, prediction):
            try:
                precision, recall, f1score, error_rate = score_card(pred, orig)
                precisions.append(precision)
                recalls.append(recall)
                f1scores.append(f1score)
                error_rates.append(error_rate)
            except:
                print('skipped')
            pbar.update(1)
 
    logging.info("[Info] Precision: {}".format(precisions))
    logging.info("[Info] Recall: {}".format(recalls))
    logging.info("[Info] F1-Score: {}".format(f1scores))
    logging.info("[Info] Error Rate: {}".format(error_rates))
    
    logging.info("[Info] Matthews Correlation Coefficient (MCC): ")
    logging.info(matthews_corrcoef(original.ravel(), prediction.ravel()))
    
    logging.info("[Info] ROC_AUC Curve: ")
    fpr, tpr, thresholds = roc_curve(original.ravel(), prediction.ravel())
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve WSCN (area = %0.4f)' % auc_score)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    plt.savefig(os.path.join(INFERENCE_DIR, "ROC_Curve_WSCN.pdf"))

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    # sys.stdout = LoggerWriter(logging.info)
    # sys.stderr = LoggerWriter(logging.error)

    main()
