# Flipkart Order Intelligence & Support Assistant

An integrated, end-to-end artificial intelligence support system developed for the Flipkart Capstone Project. This repository synthesises classical machine learning, deep learning transfer learning, and stateful agentic AI using LangGraph into a single, cohesive customer support utility.

---

## 📂 Repository Structure

```text
├── data/
│   └── sample_images/       # 5+ exported Fashion-MNIST test images (.png)
│       ├── 01_trouser.png
│       ├── 05_sandal.png
│       ├── 07_sneaker.png
│       ├── 08_bag.png
│       └── 09_ankle_boot.png
├── models/
│   ├── return_risk_model.pkl    # Part 1: Preprocessing & Random Forest pipeline
│   └── product_classifier.pt   # Part 2: Trained image categoriser weights
├── transcripts/             # Part 3: All 8+ agent test conversations (.txt)
│   ├── transcript_01_policy_return_window.txt
│   ├── transcript_02_policy_cod_refund.txt
│   ├── transcript_03_tool_return_risk.txt
│   ├── transcript_04_tool_image_classification.txt
│   ├── transcript_05_multi_turn_state.txt
│   ├── transcript_06_state_reset.txt
│   ├── transcript_07_prompt_injection.txt
│   └── transcript_08_hallucination_refusal.txt
├── .gitignore                    # Excludes virtual env, caches, and system files
├── requirements.txt              # Pre-configured project dependencies
├── knowledge_base.json           # Part 3: The 12 Flipkart customer policies
├── generate_orders.py            # Part 1: Seeded dataset generator script
├── orders_dataset.csv            # Part 1: Generated orders dataset (6,000 rows)
├── part1_complete_pipeline.py    # Part 1: Comprehensive ML training and evaluation
├── part1_visualizer.py           # Part 1: Matplotlib analysis plotting script
├── part2_img_classifier_train.py # Part 2: CNN transfer learning and image export script
├── part3_tools.py                # Part 3: Active prediction & classification tool functions
├── part3_rag_eval-v2.py          # Part 3: Neural FAISS RAG Index evaluation script
├── part3_agent-v2.py             # Part 3: Stateful support agent graph code
└── README.md                     # This file (Project documentation)
```

---

## 🚀 Setup & Execution Guide

### **1. Environment Setup**
Ensure Python 3.10+ is installed. Clone this repository, activate your virtual environment, and install all required local, free-tier dependencies:
```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Part 1: Dataset Generation, Training & Plotting**
To regenerate the deterministic dataset, run the complete training pipeline, and plot analytics:
```bash
# Generate and verify dataset, train models, and export 'return_risk_model.pkl'
python part1_complete_pipeline.py

# Generate decision sweep and feature importance Gini vs. Permutation plots
python part1_visualizer.py
```

### **3. Part 2: Product Image Categoriser Training**
To download the Fashion-MNIST dataset, train the transfer learning classifier head, export `product_classifier.pt`, and generate the sample test images:
```bash
python part2_img_classifier_train.py
```

### **4. Part 3: Flipkart Support Agent & Evaluation**
The agent operates in an offline, local neural search mode by default using `all-MiniLM-L6-v2` and `FAISS`—**no external API keys are required for evaluation**.
```bash
# Run the RAG evaluation to calculate Precision@3 and Recall@3
python part3_rag_eval-v2.py

# Run the stateful agent to simulate all 8 support scenarios and export transcripts
python part3_agent-v2.py
```

---

## 📊 Part 1: Return-Risk Scoring Pipeline (35 Marks)

### **1. Dataset Verification & EDA**
*   **Total Row Count:** `6,000` rows
*   **Total Columns:** `13` features
*   **Overall Return Rate:** **`22.7500%`** (Comfortably within the 18% to 27% acceptance band)
*   **Percentage of Missing `rating_given`:** **`13.0500%`** (Comfortably within the 8% to 18% acceptance band)

#### **Subgroup Return Rates**
| Product Category | Return Rate (%) | | Payment Method | Return Rate (%) |
| :--- | :---: | :--- | :--- | :---: |
| Apparel | `26.4275%` | | COD | `30.7477%` |
| Beauty | `20.0345%` | | Prepaid_Card | `16.8154%` |
| Electronics | `18.6930%` | | Prepaid_UPI | `16.9199%` |
| Footwear | `25.9570%` | | Wallet | `17.8451%` |
| Home | `19.1469%` | | | |

---

### **2. Missingness Analysis (`rating_given`)**
The missingness pattern of `rating_given` is classified as **Missing at Random (MAR)**.

*   **Justification:** The column missingness is conditionally dependent on the observed variable `payment_method`. Specifically, looking at the dataset generation logic inside `generate_orders.py`:
    *   **COD Orders:** Missingness probability is set to `22%`.
    *   **Non-COD Orders (Prepaid/Wallet):** Missingness probability is set to `6%`.
    *   **Evidence:** In the generated dataset, the observed missingness for COD orders is **`22.8309%`** versus **`6.0589%`** for prepaid/wallet orders, creating a significant missingness gap of **`16.7720%`**. Because this missingness is systematic and fully explained by an observed variable rather than the unobserved rating value itself, it represents a classic MAR pattern.

---

### **3. Preprocessing Pipeline**
To prevent data leakage, a scikit-learn `ColumnTransformer` was used to bundle preprocessing steps:
*   **Numeric Features** (`price_inr`, `discount_pct`, `customer_tenure_days`, `num_previous_orders`, `num_previous_returns`, `delivery_distance_km`, `delivery_days`, `is_weekend_order`): Imputed with the `median` and standard-scaled via `StandardScaler()`.
*   **Categorical Features** (`product_category`, `payment_method`): Imputed with the `most_frequent` mode and one-hot encoded via `OneHotEncoder(sparse_output=False, handle_unknown="ignore")`.
*   **Leakage Avoidance:** Preprocessing transformers were strictly fitted on the 80% training split only (`fit_transform`), and subsequently applied to both the training and 20% test splits (`transform`) to guarantee no training-test leakage occurred.

---

### **4. Baseline Model Comparison**
*   **Model:** DummyClassifier (Most-Frequent Strategy) on an 80/20 stratified split.
*   **Accuracy:** **`77.2500%`**
*   **F1-Score (Class 1 - Returned):** **`0.000000`** (Zero recall, zero precision for the positive class)

> **Misleading Accuracy Trap:**
> In imbalanced e-commerce datasets where the minority class (returned orders) accounts for only ~22.75% of the data, a baseline classifier that blindly guesses "Not Returned" (0) will yield a deceptively high accuracy of **`77.25%`**. However, this classifier is completely useless for business operations because it has **zero recall**—failing to flag a single high-risk return. To solve real-world problems, we must align model evaluation to business cost structures using metrics like Class-1 F1-Score, ROC-AUC, and precision-recall trade-offs.

---

### **5. Logistic Regression Tuning & Business Trade-offs**
*   **Default (0.5 Threshold):** Accuracy: `59.1667%` | F1: `39.2060%` | Recall: `57.8755%` | Precision: `29.6435%` | ROC-AUC: `0.625263`
*   **F1-Maximising Threshold (\\(t^*_{lr}\\)):** **`0.44`**
*   **Metrics at \\(t^*_{lr}\\):** Accuracy: `50.1667%` | F1: **`40.9091%`** | Recall: **`75.8242%`** | Precision: `28.0108%`

> **Business Trade-off Analysis:**
> Shifting the decision threshold down from 0.5 to `0.44` represents a deliberate business choice. By lowering the threshold, we accept more false positives (orders flagged as high return-risk that are actually kept) to capture substantially more true positives (catching returns before they occur). The premium of carrying out proactive intervention (e.g., confirming orders via support, tightening verification checks) is much lower than the actual logistics and processing costs of handling physical returns. Hence, we accept a minor precision drop of `-1.6327%` to boost recall by **`+17.9487%`** percentage points (raising recall from `57.88%` to `75.82%`).

---

### **6. Random Forest Model & GridSearchCV**
*   **GridSearchCV Parameters:** `n_estimators` ∈ [100], `max_depth` ∈ [6, 10, None], evaluated with 5-Fold StratifiedKFold cross-validation scored on `roc_auc`.
*   **Winning Hyperparameters:** `n_estimators`: `100` | `max_depth`: `6`
*   **Best Cross-Validated ROC-AUC:** **`0.617836`**
*   **Held-out Test-set ROC-AUC:** **`0.614286`** (Within `0.003` of the CV score, demonstrating stable generalization).

---

### **7. Feature Importance Analysis**

Below is the side-by-side comparison of the Gini (impurity-based) feature importances from decision tree splits versus the Permutation Feature importances calculated on the held-out test split:

| Rank | Impurity-Based Importance (`.feature_importances_`) | Permutation Importance (On Test Set) |
| :---: | :--- | :--- |
| **1** | `payment_method_COD` (0.16646) | `payment_method_COD` (0.06894) |
| **2** | `price_inr` (0.13711) | `price_inr` (0.00801) |
| **3** | `customer_tenure_days` (0.10743) | `payment_method_Prepaid_UPI` (0.00767) |
| **4** | `delivery_distance_km` (0.09724) | `num_previous_returns` (0.00711) |
| **5** | `discount_pct` (0.08901) | `product_category_Home` (0.00357) |

> **Permutation vs. Gini Impurity Comparison:**
> Gini impurity-based feature importance evaluates features by how frequently tree splits are performed on them. This creates a severe bias towards high-cardinality, continuous variables like `customer_tenure_days` (Rank 3), `delivery_distance_km` (Rank 4), and `discount_pct` (Rank 5). When we apply Permutation Feature Importance—which shuffles values on the test set and measures the drop in ROC-AUC—these high-cardinality columns collapse to ranks **18, 15, and 16** respectively, with mean importances near or below zero. Meanwhile, `payment_method_COD` robustly holds Rank 1. This proves Gini importances overrate continuous training noise that provides many splitting boundaries but does not generalize to unseen test data.

---

### **8. Subgroup Analysis & Corrective Proposals**
*   **Weakest Category Subgroup (Lowest Precision):** `Home` — Precision: `22.0183%` | Recall: `70.5882%`
*   **Weakest Payment Subgroup (Catastrophic Recall):** `Prepaid_Card` — Precision: `13.0435%` | Recall: **`6.1224%`**

> **Root-Cause Subgroup Analysis and Proposal:**
> The subgroup analysis reveals a critical performance disparity across payment methods: the model achieves an exceptional **`98.06%` Recall on Cash On Delivery (COD) orders**, but fails catastrophically on prepaid channels, showing a recall of only **`6.12%` on Prepaid_Card**, **`14.29%` on Wallet**, and **`16.67%` on Prepaid_UPI**. This disparity arises because the ground-truth data-generating function treats COD as an overwhelmingly strong driver of returns, making the random forest heavily dependent on the `payment_method_COD` indicator to trigger a high-risk score. For prepaid transactions, where this indicator is absent, the model struggles to detect risk.
> 
> **Actionable Corrective Proposal:** Rather than using a unified decision threshold, Flipkart should implement payment-specific decision thresholds. Since prepaid orders have lower baseline return rates (`~17%` vs. `~31%` for COD) and the risk scores generated by the Random Forest are compressed into a much lower probability scale, a unified threshold of `0.46` is too aggressive. We propose lowering the decision threshold to **`0.15` for Prepaid Card/UPI orders**. This adjustment would expand the model's sensitivity specifically for non-COD orders, catching high-risk returns in prepaid transactions without affecting the high-performing COD pipeline.

---

### **9. Saved Pipeline Artifact**
*   **Model Location:** `models/return_risk_model.pkl` (A unified scikit-learn Pipeline containing both preprocessing and the tuned Random Forest classifier).
*   **F1-Maximising Random Forest Threshold (\\(t^*_{rf}\\)):** **`0.46`** (Yields overall test F1-score of **`0.396181`**, Recall of `60.8059%`, and Precision of `29.3805%`).

---

## 👟 Part 2: Product Image Categoriser via Transfer Learning (25 Marks)

### **1. Training Configuration & Splits**
*   **Dataset:** Fashion-MNIST (10 apparel/footwear categories)
*   **Split Sizes:** Training split: `55,000` | Stratified Validation split: `5,000` | Held-out Test split: `10,000`
*   **Pretrained Backbone:** `ResNet-18`
*   **Hyperparameters:** Batch Size: `128` | Optimizer: `Adam` | Learning Rate: `0.001` | Epochs: `15`
*   **Preprocess Transforms:** Resize to `64x64`, channel replication (1 to 3 grayscale duplication), and standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### **2. Training Strategy & Optimization**
*   **Feature Caching Advantage:** Because the ResNet-18 backbone was frozen, we ran a single feature-extraction pass over the entire dataset and cached the output `512-dimensional` activation features directly in RAM. This avoided re-running the frozen CNN layers during every epoch, accelerating head training to **under 15 seconds**!
*   **Final Held-Out Test Accuracy:** **`89.5500%`** (Comfortably sailing past the `80%` minimum validation threshold).

### **3. Visual Confusion Analysis**

The transfer-learning model excels at classifying highly distinctive silhouettes, achieving exceptional F1-scores on **Trouser (`0.97`)**, **Sandal (`0.97`)**, and **Bag (`0.97`)**. However, it encounters systematic visual confusion between specific garments:

#### **Shirt (Class 6) vs. T-shirt/top (Class 0)**
The F1-score drops to its lowest at **`0.70`**. The confusion matrix shows that approximately **`18% to 22%` of actual Shirts are misclassified as T-shirts/tops**.
*   **Visual Reason:** Shirts and T-shirts share an almost identical primary silhouette (torso, shoulder cuts, neck opening). At 28x28 grayscale resolution, fine spatial details (collars, buttons, plackets) are compressed and blurred, making them indistinguishable for early feature extractors.

#### **Pullover (Class 2) vs. Coat (Class 4)**
The F1-score is constrained by a **`12%` misclassification rate** of Pullovers being mislabeled as Coats.
*   **Visual Reason:** Both represent long-sleeve, torso-covering outer layers. Grayscale contour maps struggle to distinguish open/zipped jacket lapels on coats from crewneck/turtleneck collars on pullovers, resulting in high feature space overlap.

### **4. Exported Sample Images**
The following sample test-set images have been exported as actual `.png` files under `data/sample_images/` to verify Part 3's visual tool:
1.  `data/sample_images/01_trouser.png` (True Category: Trouser)
2.  `data/sample_images/05_sandal.png` (True Category: Sandal)
3.  `data/sample_images/07_sneaker.png` (True Category: Sneaker)
4.  `data/sample_images/08_bag.png` (True Category: Bag)
5.  `data/sample_images/09_ankle_boot.png` (True Category: Ankle boot)

---

## 🤖 Part 3: Flipkart Support Agent (40 Marks)

### **1. Knowledge Base & Vector Index Setup**
*   **Content:** Written database containing 12 distinct policy documents (`knowledge_base.json`) covering Return Windows, COD refund timelines, delivery SLAs, and reverse-pickup eligibility.
*   **Chunking Strategy:** Chunked sentence-wise to preserve conversational granularity.
*   **Embedding Model:** `all-MiniLM-L6-v2` (free, offline, 384 dimensions) via `sentence-transformers`
*   **Vector Database:** Local `FAISS` index (`faiss.IndexFlatL2`) measuring dense Euclidean distances.

### **2. Integrated Support Tools**
1.  `check_return_risk(order_features: dict) -> dict`
    *   Loads the Random Forest pipeline from `models/return_risk_model.pkl`.
    *   **Calibrated Risk Buckets:** To ensure robust self-calibration across hyperparameter variations, risk categories are anchored directly to the optimal threshold \\(t^*_{rf} = 0.46\\):
        *   **Low Risk:** Probability < `0.46` (Approve standard processing).
        *   **Medium Risk:** `0.46` <= Probability < `0.61` (where the upper limit is \\(t^*_{rf} + 0.15\\)—Schedule callback).
        *   **High Risk:** Probability >= `0.61` (Flag for customer-relations review).
2.  `classify_product_image(image_path: str) -> dict`
    *   Loads `models/product_classifier.pt` and runs PyTorch inference on any `.png` file in `data/sample_images/`.

### **3. Stateful LangGraph Architecture**
The agent is designed using a branching state machine in LangGraph:
```text
           [Start]
              │
       [1. Intent Router] ────(Conditional Edge)────┐
         /          \                             │
[2. RAG Retrieval]   [3. Tool Calling Node]        │
        \\            /                            │
     [4. Response Generator] ◄────────────────────┘
              │
            [End]
```
*   **Conversational State Management:** Incorporates multi-turn state carrying. Conversation history stores state variables (such as `user_name` or `chat_history`) and is verified to reset to empty on a fresh thread initialization.

### **4. Guardrails & Deflections**
*   **Input-Side Prompt-Injection Filtering:** Intercepts system override keywords (e.g., "ignore previous instructions") and cleanly deflects, refusing to comply with prompt hijacking.
*   **Output-Side Groundedness Check:** Computes vector distances during FAISS retrieval. If the distance score of the best retrieved chunk is above **`1.15`** (indicating low similarity), the agent halts and gracefully declines to answer, preventing factual hallucinations.

---

### **5. Retrieval Evaluation**

| Query | Retrieved Chunks | Relevant Documents | Precision@3 | Recall@3 |
| :--- | :---: | :---: | :---: | :---: |
| **Q1:** `I ordered a pair of shoes but they are too tight...` | `Doc 1, Doc 10, Doc 6` | `Doc 1` | `1/3` | `1/1` |
| **Q2:** `I paid for my order in cash at the door...` | `Doc 5, Doc 12, Doc 6` | `Doc 5` | `1/3` | `1/1` |
| **Q3:** `I live in New Delhi. How fast will my orders...` | `Doc 7, Doc 5, Doc 3` | `Doc 7` | `1/3` | `1/1` |
| **Q4:** `Does the delivery person pack the shirt...` | `Doc 10, Doc 9, Doc 11` | `Doc 9, Doc 10` | `2/3` | `2/2` |
| **Q5:** `My laptop arrived with a cracked screen...` | `Doc 5, Doc 2, Doc 6` | `Doc 2` | `1/3` | `1/1` |
| **Average** | — | — | **`40.0000%`** | **`100.0000%`** |

*   *Note: Arithmetic is evaluated at the document level. If multiple chunks from the same document are retrieved in the top 3, they are deduplicated before scoring.*

---

## 💬 Conversation Transcripts

All required conversation transcripts are recorded and linked below.
*(See detailed files in the `transcripts/` folder)*

1.  **[Transcript A: RAG Policy Query 1](transcripts/transcript_01_policy_return_window.txt):** Customer asking about apparel return windows.
2.  **[Transcript B: RAG Policy Query 2](transcripts/transcript_02_policy_cod_refund.txt):** Customer asking about COD refund timelines.
3.  **[Transcript C: Return Risk Prediction Tool](transcripts/transcript_03_tool_return_risk.txt):** Evaluating a customer's high-risk order using `check_return_risk`.
4.  **[Transcript D: Image Classification Tool](transcripts/transcript_04_tool_image_classification.txt):** Correctly classifying an accessory bag image from `data/sample_images/07_sneaker.png`.
5.  **[Transcript E: Multi-Turn Conversation State](transcripts/transcript_05_multi_turn_state.txt):** Customer introducing themselves as Amit Kumar, and the agent carrying that context over.
6.  **[Transcript F: Fresh Thread State Reset](transcripts/transcript_06_state_reset.txt):** Proving that the previous conversation's state is completely absent when starting a brand new thread.
7.  **[Transcript G: Prompt Injection Deflection](transcripts/transcript_07_prompt_injection.txt):** Prompt injection attack attempted and successfully deflected by input-side filters.
8.  **[Transcript H: Groundedness Threshold Refusal](transcripts/transcript_08_hallucination_refusal.txt):** An out-of-domain query gets cleanly rejected due to low similarity, preventing factual hallucinations.

---

## 🛠️ Verification & Git Workflow Proof

To verify your repository's Git workflow compliance, run `git log --graph --oneline --all` to display your development branches. You must see:
1.  3 feature branch for each part.
2.  At least two independent commits on one of the feature branch.
3.  A merge commit integrating the feature branch back into the `main` branch.

**Example branch log:**
```text
*   86550e1 (HEAD -> Part-3-Agent-Implementation, origin/main, origin/HEAD, main) Merge pull request #3 from ashuawasthi98/Part-2-Product-Image-Categoriser-via-Transfer-Learning
|\
| * 0afcc22 (origin/Part-2-Product-Image-Categoriser-via-Transfer-Learning, Part-2-Product-Image-Categoriser-via-Transfer-Learning) Update Readme with Part2 Analysis
| * f4ac369 Part 2: Product Image Categoriser training
|/
*   6d90c3b Merge pull request #2 from ashuawasthi98/Part-1-Dataset-Generation-Return-Risk-Training
|\
| * 09e6b1c (origin/Part-1-Dataset-Generation-Return-Risk-Training, Part-1-Dataset-Generation-Return-Risk-Training) Part1 - Complete Implementation
| * b9c2c19 Implement ColumnTransformer preprocessing and train baseline DummyClassifier
```
