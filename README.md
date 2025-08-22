# Hogwarts Features Explorer

## 📖 About
This project is a small toolkit to **analyze, visualize, and model Hogwarts students’ features**.  
It has two main goals:

1. **Exploration & visualization** — understand the dataset by describing features, plotting distributions, correlations, and separability across Houses.
2. **Modeling** — train a logistic regression classifier on the training dataset and infer predictions on the test dataset.

Input datasets are CSV files with a `Hogwarts House` column and numeric features such as:

```
Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts, Divination,
Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Potions,
Care of Magical Creatures, Charms, Flying
```

---

## ⚙️ Setup

Clone the repository, create a virtual environment, and install requirements:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirement.txt
```

---

## 🚀 Usage

The project is controlled with `make`.  
Make sure you are in the project root directory and run one of the following:

```bash
# 1) Show descriptive statistics (like pandas .describe but custom)
make describe

# 2) Generate plots:
#    - most homogeneous feature across houses (histogram)
#    - most correlated feature pair (scatter)
#    - pairwise scatter matrix + ANOVA F-scores
make plot

# 3) Train the logistic regression model on dataset_train.csv
make train

# 4) Run inference with the trained model on dataset_test.csv
make infer

# 5) Remove temporary files (saved numpy arrays)
make clean
```

---

## 📊 Algorithms and methods

### Descriptive statistics (`make describe`)
- Custom implementation of statistics (count, mean, std, min, max, percentiles, IQR, range, coefficient of variation, MAD).  
- Implemented manually without relying on pandas `.describe()`.

### Histogram analysis (`histogram.py`)
- Computes a **homogeneity score** for each feature:  
  - Bin feature values (`np.histogram`).  
  - Build density histograms per House.  
  - Compare houses via average pairwise distance.  
  - Lower score = more homogeneous.  
- Selects the most homogeneous feature and plots per-House histograms.

### Scatter correlation (`scatter_plot.py`)
- Computes the **Pearson correlation matrix**.  
- Finds the feature pair with the **strongest absolute correlation**.  
- Plots their scatter with the correlation value.

### Pairwise scatter + ANOVA (`pair_plot.py`)
- Builds a **scatter matrix** (`pandas.plotting.scatter_matrix`) colored by House.  
- Computes **ANOVA F-scores** (`sklearn.feature_selection.f_classif`) to rank features by how well they separate Houses.  
- Prints the ranking and suggests the top features to keep for logistic regression.

### Model training & inference
- **Logistic regression** (implemented from scratch, not scikit-learn’s LogisticRegression).  
- `mainTrain.py`: trains weights on `datasets/dataset_train.csv`, saves parameters as `.npy`.  
- `mainInfer.py`: loads trained parameters, predicts Houses for `datasets/dataset_test.csv`.

---

## Summary

With just `make describe`, `make plot`, `make train`, and `make infer` you can fully explore the dataset and build a classifier.