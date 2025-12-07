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
