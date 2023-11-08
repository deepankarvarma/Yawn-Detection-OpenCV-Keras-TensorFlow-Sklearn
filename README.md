# Yawning Detection Model

This repository contains Python code for generating a yawning detection model and utilizing it to detect yawning from a live camera stream. The model architecture is defined as follows:

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## Dataset

The dataset used for training and evaluation can be downloaded from Kaggle: [Yawn Dataset](https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset). It contains labeled images specifically curated for yawning detection.

## Dependencies

To run the code in this repository, you'll need the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- OpenCV

You can install the required packages using `pip`:

```shell
pip install tensorflow keras opencv-python
```

## Usage

1. Clone this repository to your local machine:

```shell
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Download the Yawn Dataset from the provided link and place it in the appropriate directory.

3. Use the provided code to train the yawning detection model.

4. Run the script to detect yawning from a live camera stream:

```shell
python predict_live.py
```

Make sure you have a camera connected to your machine for the live stream.

## Results

The yawning detection model, trained on the Yawn Dataset, can accurately identify yawning instances from a live camera stream. You can modify the code and experiment with different architectures or techniques to potentially enhance the performance.

## Acknowledgments

- The Yawn Dataset used in this project was sourced from Kaggle: [Yawn Dataset](https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset).

## License

This project is licensed under the [MIT License](LICENSE).

## This repository code for model generation and detects yawning from live camera stream.

### Dataset Download Link :- https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset
