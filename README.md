# AI/ML Learning Repository

## Python Foundations

Read, explore, manipulate, and visualize data to tell stories, solve business
problems, and deliver actionable insights and business recommendations using
some of the most widely used Python packages.
• Python Programming Fundamentals (Variables and Datatypes, Data
Structures, Conditional and Looping Statements, Functions)
• Python for Data Science - NumPy and Pandas
• Python for Visualization
• Exploratory Data Analysis (Univariate Analysis, Bivariate/Multivariate
Analysis, Missing Value Treatment, Outlier Detection and Treatment)
• AI Application Case Studies

## Machine Learning

Understand the concept of learning from data, build linear and non-linear
models to capture the relationships between attributes and a known outcome,
and discover patterns and segment data with no labels.
Graded Project | Personal Loan Campaign Purchase Prediction
Build a Machine Learning model to identify potential customers for a bank
who have a higher probability of purchasing the loan and the driving
factors behind the decision making.
• Intro to Supervised Learning - Linear Regression
• Decision Trees (Regression Trees, Logistic Regression)
• K-Means Clustering (Hierarchical Clustering, Dimensionality Reduction, PCA)

## Advanced Machine Learning

Combine the decisions from multiple models using ensemble techniques to
improve model performance and make better predictions, and employ feature
engineering techniques and hyperparameter tuning to arrive at generalized,
robust models to optimize associated business costs.
• Bagging and Random Forest
• Boosting (AdaBoost, Gradient Boosting, XGBoost, Stacking)
• Model Tuning

## Enablement

Hands-on tutorials and beginner-friendly notebooks for GPU computing, deep learning frameworks, and local LLM deployment. Each subdirectory contains a comprehensive Jupyter notebook with a 100,000-record dataset for practical exercises.

### Directory Structure

```
enablement/
├── cuda/
│   ├── cuda_level_101.ipynb
│   └── cuda_sample_data.csv
├── tensorflow/
│   ├── tensorflow_level_101.ipynb
│   └── tensorflow_sample_data.csv
├── pytorch/
│   ├── pytorch_level_101.ipynb
│   └── pytorch_sample_data.csv
└── ollama/
    ├── ollama_level_101.ipynb
    └── ollama_sample_data.csv
```

### Subdirectories

#### cuda/
GPU programming with NVIDIA CUDA for parallel computing.

| File | Description |
|------|-------------|
| `cuda_level_101.ipynb` | Beginner's guide to CUDA: GPU architecture, writing kernels with Numba, memory management, and CPU vs GPU performance comparison |
| `cuda_sample_data.csv` | 100,000 records of 3D vector data (vector_a, vector_b components), scalar values, matrix indices, and intensity values for parallel processing exercises |

#### tensorflow/
Deep learning with Google's TensorFlow framework.

| File | Description |
|------|-------------|
| `tensorflow_level_101.ipynb` | Complete TensorFlow tutorial: tensors, Keras API, building neural networks, training binary classifiers, model evaluation with ROC curves and confusion matrices |
| `tensorflow_sample_data.csv` | 100,000 records for binary classification with 10 numerical features (feature_1 to feature_10), 2 categorical features (category_a, category_b), and binary labels |

#### pytorch/
Deep learning with Meta's PyTorch framework.

| File | Description |
|------|-------------|
| `pytorch_level_101.ipynb` | Complete PyTorch tutorial: tensors, autograd, custom nn.Module classes, DataLoaders, training regression models, and model evaluation with MAE/RMSE/R² metrics |
| `pytorch_sample_data.csv` | 100,000 records for house price regression with features: size_sqft, bedrooms, bathrooms, age_years, distance_downtown, lot_size, garage_spaces, quality_score, neighborhood_rating, has_pool, has_basement, and target price |

#### ollama/
Running large language models locally with Ollama.

| File | Description |
|------|-------------|
| `ollama_level_101.ipynb` | Complete Ollama tutorial: installation, API usage, Python integration, custom Modelfiles, building chatbots, Q&A systems, and embeddings for semantic search |
| `ollama_sample_data.csv` | 100,000 records of text data with fields: topic, question_type, complexity, sentiment, question, context, word_count, and char_count for NLP exercises |

## UTA Projects

Applied machine learning projects from UT Austin's AI/ML program, featuring real-world business case studies with complete end-to-end workflows including EDA, model building, evaluation, and business recommendations.

### Directory Structure

```
uta_projects/
├── README.md
├── code/
│   ├── 1-Food Hub Data Analysis.ipynb
│   ├── 2-Machine Learning notebook.ipynb
│   ├── 3-Bank_and_Credit_Churn.ipynb
│   └── 4-Case_Study_DiabetesRisk_Prediction.ipynb
└── data/
    ├── foodhub_order.csv
    ├── Loan_Modelling.csv
    ├── BankChurners.csv
    └── pima-indians-diabetes.csv
```

### Project Notebooks

#### 1. FoodHub Data Analysis
**File:** `1-Food Hub Data Analysis.ipynb`
**Dataset:** `foodhub_order.csv`

Exploratory data analysis for a NYC-based food delivery aggregator service. Analyzes order patterns, delivery times, cuisine popularity, and customer behavior.

| Aspect | Details |
|--------|---------|
| Problem Type | Exploratory Data Analysis |
| Key Metrics | Delivery time, order cost, weekend vs weekday patterns |
| Techniques | Univariate/bivariate analysis, visualization, statistical summaries |
| Business Insights | Restaurant performance, cuisine demand, delivery optimization |

#### 2. Personal Loan Campaign Modeling
**File:** `2-Machine Learning notebook.ipynb`
**Dataset:** `Loan_Modelling.csv`

AllLife Bank campaign optimization for converting liability customers to personal loan customers. Uses Decision Tree classification with pruning techniques.

| Aspect | Details |
|--------|---------|
| Problem Type | Binary Classification |
| Algorithm | Decision Tree (CART) |
| Key Features | Income, CCAvg, CD_Account, Education, Family |
| Techniques | Pre-pruning, post-pruning (cost complexity), feature importance |
| Performance | Optimized tree depth and complexity for generalization |

#### 3. Credit Card Customer Churn Prediction
**File:** `3-Bank_and_Credit_Churn.ipynb`
**Dataset:** `BankChurners.csv`

Thera Bank customer churn analysis to identify at-risk credit card customers and key drivers of attrition. Implements ensemble learning methods with class imbalance handling.

| Aspect | Details |
|--------|---------|
| Problem Type | Binary Classification (Imbalanced) |
| Algorithms | Random Forest, Gradient Boosting, XGBoost |
| Class Balancing | SMOTE, RandomUnderSampler |
| Best Model | Random Forest with RUS (93.75% accuracy, 94% recall) |
| Key Features | Total_Trans_Ct, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1 |

#### 4. Diabetes Risk Prediction
**File:** `4-Case_Study_DiabetesRisk_Prediction.ipynb`
**Dataset:** `pima-indians-diabetes.csv`

Medical diagnosis case study predicting diabetes risk using the Pima Indians Diabetes dataset. Focuses on healthcare analytics and risk factor identification.

| Aspect | Details |
|--------|---------|
| Problem Type | Binary Classification (Medical) |
| Dataset | Pima Indians Diabetes Database |
| Features | Pregnancies, Glucose, Blood Pressure, BMI, Age, etc. |
| Focus | Healthcare risk prediction, feature analysis |

### Data Files

| File | Records | Description |
|------|---------|-------------|
| `foodhub_order.csv` | ~1,900 | Food delivery orders with restaurant, cuisine, cost, and delivery time data |
| `Loan_Modelling.csv` | ~5,000 | Bank customer data with demographics, account info, and loan acceptance labels |
| `BankChurners.csv` | ~10,000 | Credit card customer data with transaction history and churn status |
| `pima-indians-diabetes.csv` | 768 | Medical diagnostic data for diabetes prediction |
