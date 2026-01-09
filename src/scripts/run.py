"""
Launcher script for the Language Identification project.

This script executes the full experimental pipeline:
1. Dataset preprocessing (filtering, splitting, feature extraction)
2. Model training
3. Evaluation on the test set

Hyperparameter tuning is available but disabled by default.

Usage:
    python -m src.scripts.run

Optional steps (uncomment as needed):
    - model.tuning()   # hyperparameter search
"""

from src.lid import LanguageIdentificationModel


def main():
    model = LanguageIdentificationModel()

    # Preprocessing: filtering, splitting, feature extraction
    model.preprocess()

    # Optional hyperparameter tuning
    # model.tuning()

    # Train the model
    model.train()

    # Evaluate on the test set
    model.test()


if __name__ == "__main__":
    main()
