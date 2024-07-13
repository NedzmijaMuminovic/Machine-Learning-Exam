# Machine-Learning-Exam
## Heart Failure Hospitalization Prediction

This project involves predicting the NYHA (New York Heart Association) classification for heart failure patients based on their medical data. The goal is to improve the model's performance by using different machine learning algorithms and handling imbalanced data.

## Installation

To run this project, you will need Python 3.x and the following libraries:

```bash
pip install pandas scikit-learn xgboost imbalanced-learn
```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/NedzmijaMuminovic/Machine-Learning-Exam
    ```

2. Navigate to the project directory:

    ```bash
    cd Machine-Learning-Exam
    ```

3. Run the script:

    ```bash
    python app.py
    ```

## Key Features

### Loading and Exploring Data
- Loading data from an Excel file (`kardiologija_hospitalizacija 2024-06-28.xlsx`).
- Displaying initial rows and basic dataset information (data types, non-null values).
- Summarizing basic statistics of numerical columns.
- Checking for missing values in each column.

### Data Cleaning
- Imputing missing values:
  - Numerical columns: Using mean values.
  - Categorical columns: Using most frequent values (mode).
- Verifying that all missing values have been filled.

### Data Preparation for Modeling
- Converting categorical column values to strings.
- Encoding categorical variables into numerical format using `LabelEncoder`.
- Splitting data into training and testing sets (80% training, 20% testing).

### Model Training (RandomForestClassifier)
- Training a `RandomForestClassifier` on the training set.
- Making predictions on the test set.

### Model Evaluation
- Evaluating the `RandomForestClassifier` model using metrics:
  - Accuracy, Precision, Recall, F1-score.
  - Generating a detailed classification report.

### Model Improvement
- Exploring another algorithm (`XGBoost`):
  - Training an `XGBClassifier`.
  - Evaluating and comparing its performance metrics with `RandomForestClassifier`.
- Handling Class Imbalance with SMOTE
  - Applying `SMOTE` (Synthetic Minority Over-sampling Technique) to balance the dataset.
  - Training a `RandomForestClassifier` on the balanced dataset.
  - Evaluating the model performance after applying `SMOTE`.

## License

This project is licensed under the MIT License.
