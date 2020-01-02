# Facial Expression Recognition
Detects Face using Haarcascades and further detects emotion in bounded face (trained a CNN emotion detector model)

## DATASET
Download dataset in this link
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Put fer2013.csv in the fer2013 folder

## To train model 

```shell
python3 train_emotion_classiffier.py
```

## To try the program on the camera
```
python3 camera_test.py
```
## To try the program on the image
```
python3 image_test "image name"
```

## Requirements
- Python 3.6.5 
- Keras : 2.2.0
- Tensorflow : 1.9.0
