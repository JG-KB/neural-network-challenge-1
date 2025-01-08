
# Student Loans with Deep Learning

This repository contains a project that uses deep learning techniques to analyze and predict student loan repayment success. The project aims to explore how machine learning models can assist in creating a recommendation system for student loans based on individual circumstances.

## Project Overview

This project focuses on the following objectives:
1. Building a neural network model to predict student loan repayment success.
2. Analyzing the impact of various features on repayment predictions.
3. Discussing the implementation of a recommendation system for student loans.

## Key Features

- **Data Preprocessing**: Includes scaling features, encoding categorical data, and preparing datasets for training and testing.
- **Deep Learning Model**: A multi-layer perceptron (MLP) model built using TensorFlow/Keras to predict binary outcomes.
- **Evaluation Metrics**: Evaluates model accuracy and loss using test data.
- **Prediction Analysis**: Outputs predictions and compares them with actual values for validation.

## Files in this Repository

- `student_loans_with_deep_learning.ipynb`: Jupyter Notebook containing the implementation of the project.
- `README.md`: This file, providing an overview of the project.

## Steps in the Project

1. **Data Preprocessing**:
   - Standardizes numerical features using `StandardScaler`.
   - Encodes categorical features for binary classification.

2. **Model Creation**:
   - Defines a neural network with:
     - Input layer based on the number of features.
     - Two hidden layers for learning patterns.
     - Output layer with a sigmoid activation function for binary classification.

3. **Model Training**:
   - Trains the model using the training dataset over 50 epochs.

4. **Model Evaluation**:
   - Calculates loss and accuracy on the test dataset.

5. **Prediction and Analysis**:
   - Makes predictions on the test data.
   - Analyzes results by comparing predictions with actual outcomes.

## Insights and Discussion

The project also explores the idea of building a recommendation system for student loans. Key points discussed include:
- Collecting relevant data such as income levels, career prospects, and financial history.
- Using context-based filtering to recommend loans tailored to individual circumstances.
- Addressing real-world challenges like economic uncertainty and student dropouts.

## How to Run the Project

1. Clone the repository or download the `student_loans_with_deep_learning.ipynb` file.
2. Open the notebook in Jupyter Notebook or Google Colab.
3. Ensure the necessary libraries (`tensorflow`, `pandas`, `sklearn`, etc.) are installed.
4. Run the notebook cells sequentially to execute the project.

## Requirements

- Python 3.8 or higher
- TensorFlow
- scikit-learn
- pandas
- numpy

## Future Work

- Expanding the dataset to include more diverse features for better predictions.
- Improving the model's architecture and hyperparameter tuning.
- Implementing a real-world recommendation system using context-based filtering.

## License

This project is licensed under the MIT License.

---

**Author**: Jaylen  
**Last Updated**: January 2025
