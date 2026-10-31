# Task 22 – Handle Missing Values (Titanic Dataset)

**Veda Technology · AI & ML Track · Level 1 · Day 22**

## Objective
Learn basic data preprocessing for machine learning by handling missing values in a real dataset using Pandas, and explain why each treatment was chosen.

## Tools Used
- Python
- Pandas
- Seaborn (to load the Titanic dataset)
- Jupyter Notebook

## Dataset
Titanic dataset (`sns.load_dataset("titanic")`) – 891 rows, 15 columns.

## Files in this Repository
| File | Description |
|------|-------------|
| `handle_missing_values.ipynb` | Notebook with analysis and cleaning code |
| `titanic_cleaned.csv` | Cleaned dataset after treatment |
| `README.md` | Approach and explanation |

## 1. Original Missing-Value Analysis

| Column | Missing Count | Missing % |
|--------|---------------|-----------|
| age | 177 | ~19.9% |
| embarked | 2 | ~0.2% |
| embark_town | 2 | ~0.2% |
| deck | 688 | ~77.2% |

All other columns had no missing values.

## 2. Treatment Applied

```python
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
df = df.drop(columns=["deck"])
```

## 3. Why These Methods Were Chosen

- **`age` → median imputation:** Age is a numerical column with a skewed distribution and some outliers. The mean gets pulled by extreme values, while the median stays stable and represents a typical passenger better.
- **`embarked` and `embark_town` → mode imputation:** These are categorical columns, so mean/median does not apply. Only 2 values were missing, so filling with the most frequent value (mode) introduces negligible bias.
- **`deck` → column dropped:** About 77% of the values were missing. Imputing that much data would mostly be guesswork and could mislead a model, so removing the column is the safer option.
- **Rows were not dropped:** Using `dropna()` on the whole dataset would have removed a large portion of the data (mainly because of `deck` and `age`). Avoiding blind removal of large portions of data preserves information for training.

## 4. Verification
After cleaning, `df.isnull().sum()` returns 0 for every column, and the final dataset is saved as `titanic_cleaned.csv`.

## 5. Key Learnings
- Always analyse the missing-value count and percentage *before* choosing a method.
- Use median for skewed numerical data, mode for categorical data.
- Drop a column only when most of its values are missing; drop rows only when few rows are affected and the missingness is random.

## How to Run
```bash
pip install pandas seaborn jupyter
jupyter notebook handle_missing_values.ipynb
```
