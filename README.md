# Flipkart Order Intelligence & Support Assistant

An integrated, end-to-end artificial intelligence support system developed for the Flipkart Capstone Project. This repository synthesises classical machine learning, deep learning transfer learning, and stateful agentic AI using LangGraph into a single, cohesive customer support utility.

---

## 📂 Repository Structure

```text
├── data/
│   └── sample_images/       # 5+ exported Fashion-MNIST test images (.png)
├── models/
│   ├── return_risk_model.pkl    # Part 1: Preprocessing & Random Forest pipeline
│   └── product_classifier.pt   # Part 2: Trained image categoriser weights
├── transcripts/             # Part 3: All 8+ agent test conversations (.txt or .md)
├── generate_orders.py       # Part 1: Seeded dataset generator script
├── orders_dataset.csv       # Part 1: Generated orders dataset (6,000 rows)
├── agent.py                 # Part 3: LangGraph agent logic and tools
├── train_risk_model.py      # Part 1: Training & evaluation script
├── train_image_model.py     # Part 2: CNN transfer learning script
└── README.md                # This file (Project documentation)
```

---

## 🚀 Setup & Execution Guide

### **1. Environment Setup**
Ensure Python 3.10+ is installed. Clone this repository and install the required local, free-tier libraries:
```bash
pip install numpy pandas scikit-learn torch torchvision pillow joblib faiss-cpu sentence-transformers langgraph
```

### **2. Part 1: Dataset Generation & Return-Risk Training**
To regenerate the dataset and train the return-risk scoring model:
```bash
# Generate the 6,000-row deterministic dataset
python generate_orders.py

# Train risk models (Logistic Regression & Random Forest) and export pipeline
python train_risk_model.py
```

### **3. Part 2: Product Image Categoriser Training**
To download the Fashion-MNIST dataset, train the transfer learning classifier head, and export sample images:
```bash
python train_image_model.py
```

### **4. Part 3: Flipkart Support Agent Execution**
The agent operates in a fully offline, deterministic mock mode by default—**no external API keys are required for evaluation**.
```bash
# Run the conversational agent in its default MOCK_LLM mode
python agent.py
```

---

## 📊 Part 1: Return-Risk Scoring Pipeline (35 Marks)

### **1. Dataset Verification & EDA**
*   **Total Row Count:** `6,000` rows
*   **Overall Return Rate:** `[INSERT_YOUR_RETURN_RATE_HERE, e.g., 22.45%]` (Expected between 18% and 27%)
*   **Percentage of Missing `rating_given`:** `[INSERT_YOUR_MISSING_PERCENTAGE_HERE, e.g., 12.30%]` (Expected between 8% and 18%)

#### **Subgroup Return Rates**
| Product Category | Return Rate (%) | | Payment Method | Return Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| Apparel | `[INSERT]%` | | COD | `[INSERT]%` |
| Electronics | `[INSERT]%` | | Prepaid_Card | `[INSERT]%` |
| Home | `[INSERT]%` | | Prepaid_UPI | `[INSERT]%` |
| Footwear | `[INSERT]%` | | Wallet | `[INSERT]%` |
| Beauty | `[INSERT]%` | | | |

### **2. Missingness Analysis (`rating_given`)**
The missingness pattern of `rating_given` is classified as **Missing at Random (MAR)**. 
*   **Justification:** The column missingness is conditionally dependent on the observed variable `payment_method`. Specifically, looking at the dataset generation logic:
    *   **COD Orders:** Missingness probability is `22%`.
    *   **Non-COD Orders (Prepaid/Wallet):** Missingness probability is `6%`.
    *   **Evidence:** In the generated dataset, the observed missing-rate gap is `[INSERT_COD_MISSING_RATE]%` for COD orders versus `[INSERT_NON_COD_MISSING_RATE]%` for prepaid/wallet orders. Because missingness is fully explained by an observed variable rather than the unobserved rating value itself, it represents a classic MAR pattern.

### **3. Preprocessing Pipeline**
To prevent data leakage, a scikit-learn `ColumnTransformer` was used to bundle preprocessing steps:
*   **Numeric Features** (`price_inr`, `discount_pct`, `customer_tenure_days`, `num_previous_orders`, `num_previous_returns`, `delivery_distance_km`, `delivery_days`): Imputed with the median and standard-scaled.
*   **Categorical Features** (`product_category`, `payment_method`): Imputed with the mode (most frequent) and one-hot encoded.
*   **Leakage Avoidance:** The pipeline was strictly fitted on the 80% training split only, and subsequently transformed both the training and 20% validation/test splits.

### **4. Baseline Model Comparison**
*   **Model:** DummyClassifier (Most-Frequent Strategy) on an 80/20 stratified split.
*   **Accuracy:** `[INSERT_DUMMY_ACCURACY, e.g., 78.50%]`
*   **F1-Score (Class 1 - Returned):** `0.0` (Zero recall, zero precision for the positive class)

> **Misleading Accuracy Trap:**
> In imbalanced e-commerce datasets where the minority class (returned orders) accounts for only ~20% of the data, a baseline classifier that blindly guesses "Not Returned" (0) will yield a deceptively high accuracy of ~80%. However, this classifier is completely useless for business operations because it has **zero recall**—failing to flag a single high-risk return. To solve real-world problems, we must align model evaluation to business cost structures using metrics like Class-1 F1-Score, ROC-AUC, and precision-recall trade-offs.

### **5. Logistic Regression Tuning & Business Trade-offs**
*   **Default (0.5 Threshold):** Accuracy: `[INSERT]`, F1: `[INSERT]`, Recall: `[INSERT]`, Precision: `[INSERT]`, ROC-AUC: `[INSERT]`
*   **F1-Maximising Threshold (\\(t^*_{lr}\\)):** `[INSERT_lr_threshold, e.g., 0.38]`
*   **Metrics at \\(t^*_{lr}\\):** Accuracy: `[INSERT]`, F1: `[INSERT]`, Recall: `[INSERT]`, Precision: `[INSERT]`

> **Business Trade-off Analysis:**
> Shifting the decision threshold down from 0.5 to `[INSERT_lr_threshold]` represents a deliberate business choice. By lowering the threshold, we accept more false positives (orders flagged as high return-risk that are actually kept) to capture substantially more true positives (catching returns before they occur). The premium of carrying out proactive intervention (e.g., confirming orders via support, tightening verification checks) is much lower than the actual logistics and processing costs of handling physical returns. Hence, we accept a precision drop of `[INSERT_PREC_DROP]%` to boost recall by `[INSERT_RECALL_GAIN]%` points.

### **6. Random Forest Model & GridSearchCV**
*   **GridSearchCV Parameters:** `n_estimators` ∈ [100, 200], `max_depth` ∈ [6, 10, None], evaluated with 5-Fold StratifiedKFold cross-validation scored on `roc_auc`.
*   **Winning Hyperparameters:** `n_estimators`: `[INSERT]`, `max_depth`: `[INSERT]`
*   **Best Cross-Validated ROC-AUC:** `[INSERT_CV_AUC, e.g., 0.695]`
*   **Held-out Test-set ROC-AUC:** `[INSERT_TEST_AUC, e.g., 0.689]` (Within 0.05 of the CV score, demonstrating stable generalization).

### **7. Feature Importance Analysis**

| Rank | Impurity-Based Importance (`.feature_importances_`) | Permutation Importance (On Test Set) |
| :--- | :--- | :--- |
| 1 | `[INSERT_IMP_1]` | `[INSERT_PERM_1]` |
| 2 | `[INSERT_IMP_2]` | `[INSERT_PERM_2]` |
| 3 | `[INSERT_IMP_3]` | `[INSERT_PERM_3]` |
| 4 | `[INSERT_IMP_4]` | `[INSERT_PERM_4]` |
| 5 | `[INSERT_IMP_5]` | `[INSERT_PERM_5]` |

> **Permutation vs. Gini Impurity Comparison:**
> `[INSERT_YOUR_INTERPRETATION_PARAGRAPH_HERE. Explain why specific features like previous return rates or tenure drive return risk. Name which features—such as delivery_distance_km—lost substantial importance under permutation, and explain in one sentence why impurity-based Gini importance overrates high-cardinality continuous columns because they provide more opportunities for random splits in tree structures.]`

### **8. Subgroup Analysis & Corrective Proposals**
*   **Weakest Category Subgroup:** `[INSERT_WEAKEST_CAT, e.g., Electronics]` — Precision: `[INSERT]%`, Recall: `[INSERT]%`
*   **Weakest Payment Subgroup:** `[INSERT_WEAKEST_PAY, e.g., COD]` — Precision: `[INSERT]%`, Recall: `[INSERT]%`

> **Concrete Proposed Action:**
> `[INSERT_YOUR_SPECIFIC_FIX_HERE. Do not write a generic 'collect more data' response. Instead, propose a concrete fix, such as implementing a category-specific risk threshold for Electronics or adding a payment-specific risk penalty.]`

### **9. Saved Pipeline Artifact**
*   **Model Location:** `models/return_risk_model.pkl` (A unified scikit-learn Pipeline containing both preprocessing and the tuned Random Forest classifier).
*   **F1-Maximising Random Forest Threshold (\\(t^*_{rf}\\)):** `[INSERT_t_rf_value]`

---

## 👟 Part 2: Product Image Categoriser via Transfer Learning (25 Marks)

### **1. Training Configuration & Splits**
*   **Dataset:** Fashion-MNIST (10 apparel/footwear categories)
*   **Split Sizes:** Training split: `55,000` | Stratified Validation split: `5,000` | Held-out Test split: `10,000`
*   **Pretrained Backbone:** `[ResNet-18 or EfficientNet-B0]`
*   **Hyperparameters:** Batch Size: `[INSERT]`, Optimizer: `Adam`, Learning Rate: `[INSERT]`, Epochs: `[INSERT]`
*   **Preprocess Transforms:** Resize to `[INSERT]x[INSERT]`, channel replication (1 to 3), and standard ImageNet normalization.

### **2. Training Strategy & Optimization**
*   **Feature Caching Advantage:** Because the backbone was frozen, we ran a single feature-extraction pass over the entire dataset and cached the output features. This avoided re-running the frozen CNN layers during every epoch.
*   **Before/After Validation Accuracy:** 
    *   Feature Extraction validation accuracy: `[INSERT_FE_ACC]%`
    *   Fine-Tuning validation accuracy: `[INSERT_FT_ACC]%` (Unfrozen late backbone layers)
*   **Final Held-Out Test Accuracy:** `[INSERT_TEST_ACC]%` (Must be ≥ 80%)

### **3. Visual Confusion Analysis**
Your confusion matrix output must be printed out here (10x10). Based on the actual confusion matrix, the model most frequently confuses the following pairs:

#### **Pair 1: [Category A] vs. [Category B]**
`[INSERT_YOUR_FIRST_ANALYSIS_PARAGRAPH_HERE. Explain in detail the visual and silhouette similarities between these two categories—e.g. Pullovers vs. Coats—and why the grayscale 28x28 resolution makes it challenging for the CNN backbone to distinguish them.]`

#### **Pair 2: [Category C] vs. [Category D]**
`[INSERT_YOUR_SECOND_ANALYSIS_PARAGRAPH_HERE. Detail the specific visual overlap between these categories—e.g., Sandals vs. Sneakers—and explain why minor pixel shifts or shape similarities lead to classification errors.]`

### **4. Exported Sample Images**
The following sample test-set images have been exported as actual `.png` files under `data/sample_images/` to verify Part 3's visual tool:
1.  `data/sample_images/01_apparel_tshirt.png` (True Category: T-shirt/top)
2.  `data/sample_images/02_footwear_ankleboot.png` (True Category: Ankle boot)
3.  `data/sample_images/03_footwear_sneaker.png` (True Category: Sneaker)
4.  `data/sample_images/04_apparel_dress.png` (True Category: Dress)
5.  `data/sample_images/05_accessory_bag.png` (True Category: Bag)

---

## 🤖 Part 3: Flipkart Support Agent (40 Marks)

### **1. Knowledge Base & Vector Index Setup**
*   **Content:** Written database containing 12 distinct policy documents covering Return Windows, COD refund timelines, delivery SLAs, and reverse-pickup eligibility.
*   **Chunking Strategy:** Chunked sentence-wise to preserve conversational granularity.
*   **Embedding Model:** `all-MiniLM-L6-v2` (free, offline, 384 dimensions)
*   **Vector Database:** Local `[FAISS / Chroma]` index.

### **2. Integrated Support Tools**
1.  `check_return_risk(order_features: dict) -> dict`
    *   Loads the Random Forest pipeline from `models/return_risk_model.pkl`.
    *   **Calibrated Risk Buckets:** To ensure robust self-calibration across hyperparameter variations, risk categories are anchored directly to the optimal threshold \\(t^*_{rf} = [INSERT\_t\_rf]\\):
        *   **Low Risk:** Probability < \\(t^*_{rf}\\)
        *   **High Risk:** Probability >= \\(t^*_{rf} + 0.15\\)
        *   **Medium Risk:** All probabilities in between.
2.  `classify_product_image(image_path: str) -> dict`
    *   Loads `models/product_classifier.pt` and runs inference on any `.png` file in `data/sample_images/`.

### **3. Stateful LangGraph Architecture**
The agent is designed using a branching state machine in LangGraph:
```text
           [Start]
              │
       [1. Intent Router] ────(Conditional Edge)────┐
         /          \                             │
[2. RAG Retrieval]   [3. Tool Calling Node]        │
        \            /                            │
     [4. Response Generator] ◄────────────────────┘
              │
            [End]
```
*   **Conversational State Management:** The state stores conversation history and contextual variables (such as active order IDs or product paths). Conversation history carries state across multi-turn queries and is verified to reset to empty on a fresh thread initialization.

### **4. Guardrails & Deflections**
*   **Input-Side Prompt-Injection Filtering:** Intercepts system override keywords (e.g., "ignore previous instructions") and cleanly deflects, refusing to comply with prompt hijacking.
*   **Output-Side Groundedness Check:** Computes vector similarity scores during retrieval. If the similarity score of the best retrieved chunk is below `[INSERT_YOUR_SIM_THRESHOLD, e.g., 0.45]`, the agent halts and gracefully declines to answer, preventing factual hallucinations.

### **5. Retrieval Evaluation**

| Query | Retrieved Chunks | Relevant Documents | Precision@3 | Recall@3 |
| :--- | :--- | :--- | :--- | :--- |
| **Q1:** `[Insert]` | `[Insert]` | `[Insert]` | `[Num]/3` | `[Num]/[Total]` |
| **Q2:** `[Insert]` | `[Insert]` | `[Insert]` | `[Num]/3` | `[Num]/[Total]` |
| **Q3:** `[Insert]` | `[Insert]` | `[Insert]` | `[Num]/3` | `[Num]/[Total]` |
| **Q4:** `[Insert]` | `[Insert]` | `[Insert]` | `[Num]/3` | `[Num]/[Total]` |
| **Q5:** `[Insert]` | `[Insert]` | `[Insert]` | `[Num]/3` | `[Num]/[Total]` |
| **Average** | — | — | **`[INSERT_AVG_PREC]`** | **`[INSERT_AVG_REC]`** |

*   *Note: Arithmetic is evaluated at the document level. If multiple chunks from the same document are retrieved in the top 3, they are deduplicated before scoring.*

---

## 💬 Conversation Transcripts

All required conversation transcripts are recorded and linked below.
*(See detailed files in the `transcripts/` folder)*

1.  **[Transcript A: RAG Policy Query 1](transcripts/01_rag_policy_1.txt):** Customer asking about apparel return windows.
2.  **[Transcript B: RAG Policy Query 2](transcripts/02_rag_policy_2.txt):** Customer asking about COD refund timelines.
3.  **[Transcript C: Return Risk Prediction Tool](transcripts/03_risk_tool_call.txt):** Evaluating a customer's high-risk order using `check_return_risk`.
4.  **[Transcript D: Image Classification Tool](transcripts/04_image_tool_call.txt):** Correctly classifying an accessory bag image from `data/sample_images/05_accessory_bag.png`.
5.  **[Transcript E: Multi-Turn Conversation State](transcripts/05_multi_turn_state.txt):** Customer referencing an order ID mentioned in an earlier turn and the agent carrying that context over.
6.  **[Transcript F: Fresh Thread State Reset](transcripts/06_fresh_thread_reset.txt):** Proving that the previous conversation's state is completely absent when starting a brand new thread.
7.  **[Transcript G: Prompt Injection Deflection](transcripts/07_injection_deflected.txt):** Prompt injection attack attempted and successfully deflected by input-side filters.
8.  **[Transcript H: Groundedness Threshold Refusal](transcripts/08_unrelated_refusal.txt):** An out-of-domain query gets cleanly rejected due to low similarity, showing the similarity score and the rejection threshold.

---

## 🛠️ Verification & Git Workflow Proof

To verify your repository's Git workflow compliance, run `git log --graph --oneline --all` to display your development branches. You must see:
1.  An active development or feature branch.
2.  At least two independent commits on that feature branch.
3.  A merge commit integrating the feature branch back into the `main` branch.

**Example branch log:**
```text
*   d5e89a2 (HEAD -> main, origin/main) Merge branch 'development' into main
|\  
| * a1c2b3e (development) Part 3: Implement LangGraph conversational agent and write 12 policy docs
| * f5b6c7a (development) Part 2: Train Fashion-MNIST Transfer Learning and save product_classifier.pt
| * e9d8c7b (development) Part 1: Complete Return-Risk Scoring Pipeline and save model
|/  
* 2b3c4d5 Initial repository structure and dataset generation seed script
```
