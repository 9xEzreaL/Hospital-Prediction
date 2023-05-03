## Hospital Cost and Frequency Prediction 

Data from NTU hospital, this project based on Deep Learning, Regression method. Trying to predict next year cost and 
frequency on hospital by previous data and some features.

## Data
**Binary features:**
- 13 kinds of binary features only 0 or 1. eg: hypertension, stroke...

**Numeric features:**
- Cost of last year(or previous years) 
- Frequency to go to hospital of last year(or previous years)

**Predict objective**
- Cost of next year (Numeric)
- Frequency to go to hospital of next year (Numeric)

## Model 
1. Regression (XGBoost)
2. Simple DNN (Feature embedding and Linear)
3. Simple RNN (LSTM)

## Training
**Environment settings**
1. create environment
    - create environment python=3.6
    - pip install -r requiremetns.txt

**Data preparation**
2. All features saved on a csv file 
    - Ex: ID, feature_a, feature_b.....label_a, label_b

**Config root**
3. Go to ```utils/configurations.py``` 
    - Fill root + meta_csv will be your meta_csv path
    - save_to will be your trained model destination

**Model training**
4. call train.py, some args maybe used:
    - --exp :your experiment name
    - --net :network (rnn, dnn, xgb)
    - --lr :learning rate
    - --num_classes :number of classes your prediction
   

```CUDA_VISIBLE_DEVICES=0 python train.py --exp first --net rnn --lr 0.002 --num_classes 2 --num_features 15```


**Result**

Acc definition: 

    correct : 0.9 < pred / label < 1.1
    incorrect : others
    Acc = n(correct) / n(correct) + n(incorrecet)

- XGBoost : 
  - frequency acc: 0.0938
  - cost acc: 0.1347
- DNN model : 
  - frequency acc: 0.9
  - cost acc: 0.45
- RNN model :
  - frequency acc: 
  - cost acc: 
- Attention model :
  - frequency acc: 0.8
  - cost acc: 0.23