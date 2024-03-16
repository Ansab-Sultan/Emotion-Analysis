# Emotion Detection with LSTM Model

## Overview

This repository hosts a machine learning project focused on emotion detection in textual data using a Long Short-Term Memory (LSTM) neural network model. The project encompasses data preprocessing, exploratory data analysis (EDA), model training, and a Streamlit web application for real-time emotion prediction.

### Components

- **analysis.ipynb**: Jupyter Notebook containing comprehensive data analysis, preprocessing steps, and LSTM model training.
- **analysis_app.py**: Python script for a Streamlit web application that allows users to input text and predicts the associated emotion.
- **data/**: Directory containing the dataset used for training and analysis.
- **models/**: Folder storing the trained LSTM model (`model.keras`) and tokenizer (`tokenizer.pickle`).
- **requirements.txt**: File listing all Python libraries and dependencies required to run the project.
- **LICENSE**: License file governing the use and distribution of the project.

## Usage

To utilize the Streamlit application locally:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the necessary dependencies by running `pip install -r requirements.txt`.
4. Execute the Streamlit app using `streamlit run analysis_app.py`.
5. Access the web application through the provided URL in the terminal.

For detailed insights into data preprocessing, model training, and analysis, refer to the `analysis.ipynb` notebook.

## Dataset

The dataset used for this project resides in the `dataset/` folder. It comprises textual data labeled with various emotions, including sadness, joy, love, anger, fear, and surprise.

## Model

The emotion detection model is built using TensorFlow and Keras, employing an LSTM architecture. The trained model is (`model.keras`) and tokenizer for text preprocessing is (`tokenizer.pickle`).

## Contributors

- [Your Name](link-to-your-github-profile) - Project Lead & Developer

Contributions to this project are welcomed through issue reporting and pull requests. Your feedback and enhancements are highly valued.

## License

This project is licensed under the [MIT License](LICENSE).
