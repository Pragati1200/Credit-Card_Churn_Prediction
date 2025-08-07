# Git Flow - Credit Card Churn Prediction Project (EXL Capstone)

This document outlines the Git branching and version control strategy for the Capstone project: Credit Card Churn Prediction.

# Branching Strategy

The repository follows a structured Git Flow suited for a solo ML project with AWS integration. The following branches should be used:

# `main`
- Final production-ready code.
- Should always contain clean, tested, and deployable code.
- Only merge from `dev` when the project is ready for final submission.

# `dev`
- Primary working branch.
- Integrate all features here before promoting to `main`.

# Feature Branches
Use dedicated branches for different components of the project:

| Branch Name      | Purpose                        |
|------------------|--------------------------------|
| `feature/eda`    | Exploratory Data Analysis, visualizations |
| `feature/ml`     | Model training, evaluation, tuning |
| `feature/aws`    | AWS integration (Aurora MySQL, EC2 scripts) |
| `presentation`   | PPT slides and related documentation |

Create additional feature branches as needed using the pattern:  
