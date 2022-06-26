## Source Code:

### Training

change the hyperparameters and configuration parameters according to need in ```src/config.py```.

To train aclnet, Run following command from ```/src``` directory.

```python models/train_model.py``` 

All the trained checkpoints for pre-training as well as full model training will be saved in ```/weights.```

Above command will train aclnet for given number of epochs in ```src/config.py```.

### Test performance

To test aclnet with trained model, Run following command from ```/src``` directory.

```python models/test_model.py ``` 

Above command will generate Precision, Recall, F1-Score, Error rate, Matthews Correlation Coefficient (MCC) and ROC AUC Curve. ROC-AUC Curve will be saved in ```inference/``` directory.