# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Name: Census Income Prediction Model
Model Version: 1.0
Model Type: Classification model (e.g., Decision Tree, Random Forest, Logistic Regression, etc.) in this case logistic regression.
Developers: Roland Schneck
Model Description: This model is designed to predict whether an individual's income exceeds $50,000 per year based on demographic and employment-related features.

## Intended Use
Primary Use Case: Research and analysis of socioeconomic factors.
Users: Social scientists, economists, data scientists, and policy makers.
Usage Scenarios: Analysis of income distribution, identification of factors influencing income, and support in developing measures for income equity.

## Training Data
Data Source: UCI Machine Learning Repository, specifically the "Census Income" dataset.
Time Frame: Data from the 1994 census.
Features: Age, work class, education level, education num, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, native country.
Number of Instances: 32,561 training instances.

## Evaluation Data
Data Source: Same as training data (split into training and test datasets).
Number of Instances: 16,281 test instances.

## Metrics
Metrics Used: Precision, Recall, fbeta
Model Performance: 0.72, 0.27, 0.39

## Ethical Considerations
Fairness: The model could reinforce unconscious biases against certain demographic groups, especially if used in decision-making processes.
Privacy: Personal data must be protected and anonymized to preserve individuals' privacy.
Transparency: The model's decision-making should be understandable and transparent to build trust among users.

## Caveats and Recommendations
Caveats: The model's predictions are based on data from 1994 and may not reflect current socioeconomic conditions.
Recommendations: Regularly update the model with newer data; conduct fairness audits to avoid discrimination; use the model as a supportive tool, not as the sole basis for decision-making.