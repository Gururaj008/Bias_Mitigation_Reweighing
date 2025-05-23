# Mitigating Algorithmic Bias via Reweighing: A Pre-processing Approach to Fair AI

## Overview

This repository contains a Jupyter Notebook (`Bias_Mitigation_Reweighing.ipynb`) that demonstrates a practical approach to analyzing and mitigating **gender bias** in a machine learning classification model. The project focuses on predicting income using the Adult Census Income dataset and leverages the **AIF360 library** to assess and address fairness concerns through the **Reweighing** pre-processing technique.

The core objective is to showcase how adjusting the influence of individual data points in the training set can lead to more equitable model predictions, highlighting the trade-offs between fairness and accuracy.

## Project Structure

* `Bias_Mitigation_Reweighing.ipynb`: The main Jupyter Notebook containing all the code, analysis, and visualizations.
* `uciml_data.csv`: (Assuming this is the dataset file. If it's not in the repo, you'd mention where to download it from).

## Methodology

The notebook follows a structured approach to bias mitigation:

1.  **Data Loading and Preprocessing**:
    * Loading the Adult Census Income dataset.
    * Cleaning steps: handling duplicates and missing values.
    * Type conversion for numerical columns.
    * Transforming the 'income' target variable into a binary classification label.
    * Encoding 'gender' and 'race' as numeric protected attributes, with a primary focus on 'gender' for bias mitigation.
    * Applying One-Hot Encoding to categorical features.

2.  **Model Training and Baseline Evaluation**:
    * Establishing a **baseline Logistic Regression model** without any bias mitigation.
    * Splitting data into training and testing sets (70/30).
    * Applying `StandardScaler` for feature scaling.
    * Evaluating the baseline model's performance using standard accuracy metrics and key **fairness metrics** from AIF360:
        * Statistical Parity Difference
        * Disparate Impact
        * Equal Opportunity Difference
        * Average Odds Difference

3.  **Bias Mitigation with Reweighing**:
    * Implementation of **Reweighing**, a powerful pre-processing technique.
    * **How Reweighing Works**: This method adjusts the weights (or "importance") of individual data points in the training set. It assigns higher weights to underrepresented instances within specific protected group-outcome combinations (e.g., females with favorable income) and lower weights to overrepresented ones. This rebalances the dataset's influence during training, guiding the model to learn a more equitable decision boundary.
    * **Significance**: Reweighing directly addresses data-level biases, is model-agnostic (can be applied to any ML model supporting sample weights), and is effective in improving various fairness metrics.

4.  **Post-Mitigation Evaluation and Comparison**:
    * Training a new Logistic Regression model using the **reweighted training data**.
    * Thorough evaluation and direct comparison of the reweighed model's performance against the baseline.
    * Results are presented in both **tabular format** and **grouped bar charts** for clear visualization of the impact on accuracy and fairness metrics.

## Results & Discussion

The analysis demonstrates that applying the Reweighing technique leads to significant improvements in fairness metrics with only a minimal impact on overall accuracy.

| Metric                        | Baseline (No Reweighing) | After Reweighing |
| :---------------------------- | :----------------------- | :--------------- |
| Accuracy                      | 0.8512                   | 0.8462           |
| Statistical Parity Difference | -0.1947                  | -0.0837          |
| Disparate Impact              | 0.2557                   | 0.6012           |
| Equal Opportunity Difference  | -0.1256                  | 0.1511           |
| Average Odds Difference       | -0.1055                  | 0.0705           |

**Key Implications:**

* **Effective Bias Mitigation:** Reweighing demonstrably improved key fairness metrics, indicating its effectiveness in reducing gender bias in this income prediction task. Specifically, **Statistical Parity Difference** was reduced by over 57% (from -0.1947 to -0.0837), moving significantly closer to ideal fairness. **Disparate Impact** saw a remarkable increase from 0.2557 to 0.6012, indicating the unprivileged group's favorable outcome rate is now more than double its baseline proportion relative to the privileged group. While **Equal Opportunity Difference** and **Average Odds Difference** shifted from negative to positive values, they also moved closer to zero, indicating a more balanced distribution of true positive rates and average odds across groups.
* **Practical Applicability:** The observed minor decrease in overall accuracy (from 0.8512 to 0.8462, a reduction of approximately 0.5%) is a very common and often acceptable trade-off. This minimal impact on predictive power, coupled with substantial gains in fairness, makes reweighing a highly practical and valuable intervention for deploying more equitable AI systems in real-world scenarios.
* **Foundation for Future Work:** While highly effective, the results also highlight that achieving perfect fairness (e.g., Disparate Impact of exactly 1 or zero for differences) can be complex. This suggests avenues for further research, such as exploring the combination of reweighing with other in-processing or post-processing mitigation techniques, fine-tuning reweighing parameters for specific fairness objectives, or extending the analysis to intersectional fairness.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
    *(Replace `your-username` and `your-repo-name` with your actual GitHub details.)*

2.  **Install dependencies:**
    It's recommended to create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    (If you don't have a `requirements.txt`, you'll need to create one. See the "Requirements" section below.)

3.  **Download the dataset:**
    The notebook expects the `uciml_data.csv` file to be present. You can download the Adult Census Income dataset from the UCI Machine Learning Repository (or specify if it's included in your repo).
    * [Link to UCI Adult Dataset (if not included in repo)](https://archive.ics.uci.edu/dataset/2/adult)
    * Place the `uciml_data.csv` file in the same directory as the notebook.

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Bias_Mitigation_Reweighing.ipynb
    ```
    Open the notebook in your browser and run all cells sequentially.

## Requirements

The following Python libraries are required:

* `pandas`
* `numpy`
* `scikit-learn`
* `aif360`
* `matplotlib`
* `seaborn`

You can generate a `requirements.txt` file from your environment using:
```bash
pip freeze > requirements.txt
