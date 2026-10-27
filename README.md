# chicago-crash-analysis
# Chicago Traffic Crash Analysis

## Project Overview

This project uses machine learning to analyze traffic crashes in Chicago and identify factors associated with fatal accidents. Our goal is to understand which conditions have the greatest relationship with fatal crashes and compare different machine learning models.

The project examines factors such as speed limits, weather, lighting conditions, road conditions, driver impairment, and seatbelt usage.

## Machine Learning Models

We use three machine learning models:

1. **Random Forest:** Uses multiple decision trees to classify crashes as fatal or nonfatal.
2. **Logistic Regression:** Estimates the probability of a crash being fatal based on different features.
3. **Gradient Boosting:** Combines decision trees to improve predictions.

We compare these models using accuracy, precision, recall, and F1-score.

## Installation

### Requirements

- Python 3.12 or a compatible version
- Git
- Jupyter Notebook
- AWS credentials with permission to access the project dataset

### Download the Repository

Open a terminal and run:

```bash
git clone https://github.com/ececkenrode/chicago-crash-analysis.git
cd chicago-crash-analysis
```

### Install Required Libraries

```bash
python3 -m pip install -r requirements.txt
```

The project uses libraries including pandas, numpy, scikit-learn, boto3, matplotlib, and seaborn. Reading Parquet files from S3 also requires `pyarrow` and `s3fs`.

## Dataset

The Chicago traffic crash dataset is stored in an AWS S3 bucket:

```text
s3://chitown-crashes-api/clean/crashes.parquet
```

The dataset includes information about crash conditions, contributing factors, and whether a crash resulted in a fatality.

Access to this bucket may require AWS credentials and appropriate permissions.

## How to Run the Project

1. Download or clone the repository.
2. Install the required Python libraries.
3. Configure AWS credentials with permission to read the dataset.
4. Open the notebook:

```bash
jupyter notebook ChiCrash.ipynb
```

5. Run the relevant notebook cells to load the data, preprocess it, train the models, and evaluate their performance.

**Note:** The notebook includes earlier exploratory code and local file references. Some cells may need adjustments depending on the user's environment.

## Quick Start: Training a Small Demo

The following example trains a Random Forest model using a small synthetic dataset. It allows users to test the basic machine learning workflow without downloading the full Chicago crash dataset.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create a small example dataset
data = pd.DataFrame({
    "speed_limit": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    "impaired": [0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
    "fatal": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
})

X = data[["speed_limit", "impaired"]]
y = data["fatal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")
print("Predictions:", model.predict(X_test))
```

This example demonstrates the training process only. The synthetic data is not intended to represent real crash patterns or provide meaningful safety predictions.

## Code Documentation

The main notebook is `ChiCrash.ipynb`.

### Data Loading

Uses pandas and AWS S3 to access Chicago traffic crash records.

### Data Preprocessing

Selects crash-related features, handles missing values, converts categorical variables into numerical features, and splits the data into training and testing sets.

### Model Training

Trains Random Forest, Logistic Regression, and Gradient Boosting classifiers to predict whether crashes are fatal.

### Model Evaluation

Uses confusion matrices and classification metrics to compare model performance.

### Feature Importance

Examines which features have the strongest influence on model predictions. The project also calculates fatal crash rates and risk ratios for selected conditions.

### Crash Prediction Examples

Creates hypothetical crash scenarios and compares predictions from the three models.

## Contact

For questions about this project, contact the repository owner through GitHub:

https://github.com/ececkenrode

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.
