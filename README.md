# Spam Email Classification using Machine Learning
This project focuses on building a robust classification system to detect spam emails using various machine learning models. The dataset used is the **Spambase** dataset, which contains features extracted from emails to help classify them as spam (`1`) or not spam (`0`).

## Project Structure

├── spambase.data              # Raw dataset
├── spambase.names             # Feature and target names
├── spambase_processed.csv     # Scaled dataset
├── models/                    # ML models and evaluation
└── notebooks/                 # Jupyter notebooks

## Technologies and Libraries Used
* Python 
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* imbalanced-learn (SMOTE)
* XGBoost

## Dataset Overview
The Spambase dataset contains **57 input features** (e.g., frequency of words/characters) and a **binary target** column `target`, where:

* `1` = Spam
* `0` = Not Spam

## Key Steps

### 1. Data Loading and Preprocessing
* Extracted feature names from `.names` file.
* Loaded `.data` file and assigned appropriate column names.
* Target column added.

### 2. Feature Scaling
* Used `StandardScaler` to normalize all features (except target).
* Saved the processed data into `spambase_processed.csv`.

### 3. Class Imbalance Handling
* Visualized initial class distribution (imbalanced).
* Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance the dataset.

### 4. Model Training & Evaluation
Trained and evaluated multiple models:
* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVC)
* XGBoost

**Metrics Evaluated**:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### 5. Hyperparameter Tuning

* Used `GridSearchCV` to find best parameters for Random Forest.
* Re-trained the model using best parameters and evaluated again.

## Results (Post Tuning - Random Forest)

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.95  |
| Precision | 0.95  |
| Recall    | 0.95  |
| F1-Score  | 0.95  |

## Confusion Matrix

                Predicted
                0     1
Actual   0     533   25
         1     34    524

## How to Run
1. Clone the repository:
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier

2. Install required packages:
pip install -r requirements.txt

3. Run the Jupyter notebook to train and test models.

## Learnings
* Importance of scaling and feature preprocessing.
* Handling imbalanced data using SMOTE.
* Evaluating and tuning multiple ML models.
* Visualizing model performance using confusion matrix and classification report.

## Credits
Dataset Source: [UCI Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase)


## License
This project is licensed under the MIT License.

