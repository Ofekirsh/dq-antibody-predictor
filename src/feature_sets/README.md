# Feature Engineering Module

## OVERVIEW

This module contains feature extraction methods for processing HLA allele data. It includes:

- **One-Hot Encoding for Eplet Features**
- **Biological Feature Extraction**
- **Combined Extraction of Both Feature Sets**

Each class operates on a `pandas.DataFrame` with a predefined format and extracts meaningful features for downstream analysis.

---

## INPUT FORMAT (ID REQUIRED)

Each class expects a `pandas.DataFrame` with the following columns:

| Column Name  | Description             |
|-------------|------------------------|
| **ID**      | Unique identifier for each row (**Required**) |
| R_DQA1_1    | Recipient DQA1 allele 1 |
| R_DQB1_1    | Recipient DQA1 allele 1 |
| R_DQA1_2    | Recipient DQA1 allele 2 |
| R_DQB1_2    | Recipient DQB1 allele 2 |
| D_DQA1      | Donor DQA1 allele       |
| D_DQB1      | Donor DQB1 allele       |
| Label       | Binary label (0 or 1)   |
| Time        | Numerical time value    |

### **Important:**
- All feature extractors require the **ID** column.
- Each **ID must have exactly two rows** (representing two samples).
- If an **ID has fewer or more than two samples**, all rows associated with that ID will be removed.

### **Example Input DataFrame**
```plaintext
     ID  R_DQA1_1 R_DQB1_1 R_DQA1_2 R_DQB1_2 D_DQA1 D_DQB1  Label  Time
0  1001   05:05   03:01    03:01    03:01    05:05  03:01   1.0    10.0
1  1001   05:05   03:01    03:01    03:01    05:05  03:01   0.0    5.0
```

---

## **Feature Extraction Classes**

### **1. One-Hot Eplet Feature Extraction (`OneHotEpletFeature`)**
Encodes recipient-donor mismatches into **one-hot eplet vectors**.

#### **Processing Steps:**
1. Convert alleles to their corresponding **amino acid sequences**.
2. Rotate **DQA1 sequences by 22** and **DQB1 sequences by 31** positions to align structures.
3. Convert amino acid sequences into **one-hot eplet vectors**.
4. Compute mismatch features between recipient and donor alleles.
5. Return a **feature vector**.

#### **Output Format**
| Column Names                        | Description                    |
|--------------------------------------|--------------------------------|
| **ID**                               | Unique identifier for each row |
| 2 D 199 A, 2 G 199 T,  ...              | One-hot encoded eplet features |
| mm_DQA_Low, mm_DQB_Low, mm_DQA_High, mm_DQB_High       | Summed mismatch vectors        |
| Label                                | Binary label (0 or 1)          |
| Time                                 | Numerical time value           |

---

### **2. Biological Feature Extraction (`BiologicalFeature`)**
Extracts biological distance-based features (**PAM, EMS, G12**) for allele pairs.

#### **Processing Steps:**
1. Retrieve **distance values** from predefined **JSON files**.
2. Compute relevant **PAM, EMS, and G12 feature values**.
3. Return a feature vector.

#### **Output Format**
| Column Names            | Description           |
|-------------------------|----------------------|
| **ID**                  | Unique row identifier |
| PAM1, PAM2              | PAM distance values  |
| EMS1, EMS2              | EMS distance values  |
| G12_R1, G12_R2, G12_D   | G12 feature values   |
| Label                   | Binary label (0 or 1) |
| Time                    | Numerical time value |

---

### **3. Combined Feature Extraction (`BioEpletFeatureExtractor`)**
Extracts both **one-hot eplet** and **biological** features and merges them while ensuring alignment.

#### **Processing Steps:**
1. Compute **one-hot eplet features**.
2. Compute **biological features**.
3. Synchronize extracted features based on **ID**.
4. Remove any **unmatched rows**.

#### **Output Format**
| Column Names                        | Description                    |
|--------------------------------------|--------------------------------|
| **ID**                               | Unique identifier for each row |
| 2 D 199 A, 2 G 199 T,  ...              | One-hot encoded eplet features |
| mm_DQA_Low, mm_DQB_Low, mm_DQA_High, mm_DQB_High       | Summed mismatch vectors        |
| PAM1, PAM2                           | PAM distance values            |
| EMS1, EMS2                           | EMS distance values            |
| G12_R1, G12_R2, G12_D                | G12 feature values             |
| Label                                | Binary label (0 or 1)          |
| Time                                 | Numerical time value           |

---

## **Usage Example**

```python
from feature_engineering.one_hot_eplet import OneHotEpletFeature
from feature_engineering.biological import BiologicalFeature
from feature_engineering.bio_eplet_extractor import BioEpletFeatureExtractor

# Load DataFrame
df = pd.read_csv("data/dq_data.csv")

# Initialize extractors
one_hot_extractor = OneHotEpletFeature()
bio_extractor = BiologicalFeature()
combined_extractor = BioEpletFeatureExtractor()

# Extract features
one_hot_features = one_hot_extractor.extract_features(df, "vectors_model/one_hot_features.csv")
bio_features = bio_extractor.extract_features(df, "vectors_model/bio_features.csv")
combined_features = combined_extractor.extract_features(df, "vectors_model/combined_features.csv")
```

---

## **Notes**
- **Ensure `ID` is present** in input data; rows without it will be removed.
- Missing alleles in mappings will result in **row removal**.
- The **combined extractor** ensures consistency by aligning rows **by ID**.
- If missing keys are detected, they will be logged in `logs/missing_keys/` and should be manually added to the JSON files.
