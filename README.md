# Key Point Prediction using 5D Point Cloud Data

This repository contains the code for training and testing a model to predict keypoints using 5D point cloud data. 
The original code referenced can be found here: (https://github.com/SizheAn/MARS/tree/main)

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.7
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/) 2.2.0
- [Keras] (https://keras.io/) 2.3.0
- Matplotlib

I would advise you to use a virtual enviroment through conda

    ```bash
    conda create -n 'enviroment_name' python=3.7
    ```
## Training

To train the model, follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/Special256/RF-HAR.git
    ```

2. Navigate to the project directory:

    ```bash
    cd RF-HAR
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the training script:

    ```bash
    python MARS_model.py --dataset <dataset_path> --model_dir <path_to_save_the_model> --save_path <save_path>
    ```

    This will start the training process and save the trained model weights to a file.

## Testing

To test the trained model, follow these steps:

1. Make sure you have completed the training steps mentioned above.

2. Run the test script:

    ```bash
    python test.py --dataset <dataset_path> --model_dir <model_dir_path> --save_path <save_path>
    ```

    This will load the trained model and evaluate its performance on the test dataset.

## License

This project is licensed under the [MIT License](LICENSE).