# hand_gesture_prediction

## Dependencies
* Python3.6
* Opencv2
* pygame
* numpy
* keras
* tensorflow-gpu

## Usage
### Train model
Before you train, please check your training data's path and classes's floderName is correspond with code.

    python train.py

### Predict
It's better for you to predict in the dack background or collect new birght background and train it.

    python predict.py
### Command
You can add any new command if you need

`u` to higher Threshold Binary's quality <br>
`l` to lower Threshold Binary's quality <br>
`s` to save current image <br>
`c` change THRESH_BINARY to THRESH_BINARY_INV <br>
`q` exit the program <br>

## Dataset
This dataset is collect by `Wylie Sun` and `Chang-Yu-Han`.<br>
Please feel free and use it.<br>
[Wylie&Yuhan's hand dataset](https://drive.google.com/open?id=1TVszX9MWXKYe0H4XSW_R8kuva2MXzEPW "link")

## Demo
![demo](https://github.com/WZS666/hand_gesture_prediction/blob/master/readme_image/demo.gif)
