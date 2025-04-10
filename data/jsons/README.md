#  JSON Files for Feature Extraction

## Overview
This directory contains **preprocessed JSON files** that play a role in **biological distance calculations, allele mappings, and feature extraction** within the pipeline.
The JSON files serve three purposes:
1. **Mapping Alleles to Amino Acid Sequences**
   - Converting raw allele information into standardized protein sequences.
2. **Computing Biological Distances (EMS, PAM, G12)**
   - Quantifying mismatches between donor and recipient using established immunogenetic scoring methods.
3. **Encoding One-Hot Eplets**
   - Transforming amino acid sequences into eplet-based encodings for feature extraction.

---

##  JSON Files Description

### **1. `ems_distances.json`**

- **Description**: The **EMS3D score** reflects donor HLA immunogenicity by quantifying differences in **tertiary structure** and **surface electrostatic potential** between donor and recipient HLA molecules. More details can be found at: [HLA Algorithms - Mohorianu Lab](https://mohorianulab.org/shiny/kosmoliaptsis/HLA_Algorithms/)
   - **Usage**: Used in **`BiologicalFeatureExtractor`** and **`BioEpletFeatureExtractor`** to calculate EMS between recipient and donor alleles.
- **Key Format**:

     ```
     "DQA1*XX:XX_DQB1*XX:XX_DQA1*XX:XX_DQB1*XX:XX"
     ```
 - **Example**:
     ```json
     {
       "DQA1*06:01_DQB1*03:01_DQA1*06:01_DQB1*03:01": -3.012,
       "DQA1*01:01_DQB1*02:02_DQA1*01:01_DQB1*02:02": 0.0
     }
     ```
**Note:** The EMS values are stored as the **negative logarithm (-log)**.

---

###   **2. `pam_distances.json`**
  - **Description**: The Point Accepted Mutation (PAM) distance quantifies the evolutionary divergence between two protein sequences by estimating the number of mutations that have occurred over time.
   - **Usage**: Used in **`BiologicalFeatureExtractor`** and **`BioEpletFeatureExtractor`** to calculate PAM between recipient and donor alleles.

- **Example**:
     ```json
     {
       "DQA1*01:01_DQB1*03:01_DQA1*02:01_DQB1*03:01": -2.567,
       "DQA1*05:05_DQB1*03:03_DQA1*05:05_DQB1*03:03": 0.0
     }
     ```
**Note:** The PAM values are stored as the **negative logarithm (-log)**.


---

###   **3. `g12_classifications.json`**
  - **Description**: Stores **G12 classifications**, which determine whether an allele pair belongs to Group 1 (G1 = 0) or Group 2 (G2 = 1).
   - **Usage**: Used in **`BiologicalFeatureExtractor`** and **`BioEpletFeatureExtractor`** to classify the two allele pairs of the recipient and the allele pair of the donor.

 - **Example**:
     ```json
     {
       "DQA1*05:05_DQB1*03:01": 0,
       "DQA1*02:01_DQB1*06:03": 1
     }
     ```

---

###   **4. `allele_to_sequence_dqa.json`**
- **Description**: Maps **DQA1 alleles** to their **amino acid sequences**. The allele mappings are based on the official **[IMGT/HLA Database](https://github.com/ANHIG/IMGTHLA/blob/Latest/fasta/DQA1_prot.fasta)**.

- **Usage**: Used in **`OneHotEpletFeature`** to convert **DQA1 alleles** into amino acid sequences.

- **Example**:
     ```json
     {
       "06:01": "MSWLGAL...",
       "01:01": "TQSFDPL..."
     }
     ```

---

###   **5. `allele_to_sequence_dqb.json`**
- **Description**: Maps **DQB1 alleles** to their **amino acid sequences**. The allele mappings are based on the official **[IMGT/HLA Database](https://github.com/ANHIG/IMGTHLA/blob/Latest/fasta/DQB1_prot.fasta)**.

- **Usage**: Used in **`OneHotEpletFeature`** to convert **DQB1 alleles** into amino acid sequences.

   - **Example**:
     ```json
     {
       "03:01": "YLMQKTV...",
       "05:05": "GFTPLAS..."
     }
     ```

---

###   **6. `all_eplets.json`**
   - **Description**: Contains the **list of All DQ eplets**. The full list of eplets is based on the **[Eplet Registry Database](https://www.epregistry.com.br/databases/DQ)**.
   - **Usage**: Used in **`OneHotEpletFeature`** to convert amino acid sequences into One-Hot Eplet encoding.

   - **Example**:
     ```json
     [
       "2 D 199 A",
       "2 G 199 T",
       "3 P 9 L 37 D"
     ]
     ```

---

###   **7. `confirm_eplets.json`**
   - **Description**: Contains the **list of Confirm DQ eplets**. The full list of eplets is based on the **[Eplet Registry Database](https://www.epregistry.com.br/databases/DQ)**.
   - **Usage**: Used in **`OneHotEpletFeature`** to convert amino acid sequences into One-Hot Eplet encoding.
   - **Example**:
     ```json
     [
       "2 D 199 A",
       "3 P 9 L 37 D",
       "23 L"
     ]
     ```


---

## **Notes**
- JSON files **should not be modified manually** unless new mappings are required.
- However, if you **run the pipeline** and see in the logs that some keys are missing during feature extraction, you may need to update these files.
- The code includes a function **`save_missing_keys`** in the extractors, which logs missing keys in **`logs/missing_keys/`**.
- If you want to **avoid missing samples**, you can:
  1. Check the missing keys in the generated files inside `logs/missing_keys/`.
  2. Manually add the missing keys to the corresponding JSON files (`ems_distances.json`, `pam_distances.json`, `g12_classifications.json`, etc.).
  3. Re-run the feature extraction pipeline after updating the JSON files.

---

### **One-Hot Eplet Feature Extraction**
- The One-Hot Eplet extraction follows a **three-step process**:
  1. **Convert Alleles to Amino Acid Sequences**
     - This step uses **`dqa_to_seq.json`** and **`dqb_to_seq.json`** to map each allele (DQA1/DQB1) to its corresponding amino acid sequence.
  2. **Sequence Rotation for Alignment**
     - To align the amino acid sequences properly for eplet extraction:
       - **DQA1 sequences are rotated by 22 positions.**
       - **DQB1 sequences are rotated by 31 positions.**
     - This ensures that eplet positions are correctly mapped based on structural alignment.
  3. **Convert Amino Acid Sequences to One-Hot Eplet Representation**
     - The extracted amino acid sequences are then mapped to eplet-based encodings using **`all_eplets.json`** or **`confirm_eplet.json`**.

#### **Example of Sequence Rotation**
  - **Original sequence**: `"abcde"`
  - **After rotating by 3 positions**: `"deabc"`

This rotation step is **necessary for consistency** before extracting eplet-based features.