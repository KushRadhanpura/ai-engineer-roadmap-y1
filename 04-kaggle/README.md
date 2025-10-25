# 04-ML-GAUNTLET: House Price Prediction (Scikit-learn Mastery)

## Project Goal
To complete the "Machine Learning Core" phase of the AI Accelerated 1-Year Roadmap by implementing, comparing, and tuning a suite of classical machine learning regression models using the professional workflow tools available in Scikit-learn.

## Technology Stack & Environment
* **Operating System:** Pop!_OS Linux
* **Core Libraries:** Pandas, NumPy, Scikit-learn (Linear Regression, Ridge, Lasso, Random Forest)
* **Workflow Tools:** Scikit-learn Pipelines, ColumnTransformer, GridSearchCV (for hyperparameter tuning)
* **Version Control:** Git / GitHub

## Key Achievements & Results
* **Objective:** Predict residential house prices (SalePrice) based on 79 features.
* **Metric:** Root Mean Squared Logarithmic Error (RMSLE) - *Lower is better.*
* **Best Model:** Tuned **Random Forest Regressor**.
* **Best Parameters:** {'regressor__max_depth': 20, 'regressor__min_samples_split': 2, 'regressor__n_estimators': 50}
* **Final Tuned RMSLE Score on Held-out Test Data:** **0.1471**

## Conclusion
The project successfully demonstrated a complete ML workflow, showing that the ensemble method (Random Forest) provided the best fit for this complex, heterogenous data, outperforming linear models after extensive cross-validation and rigorous hyperparameter tuning.