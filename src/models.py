from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_model(model_type="rf"):
    if model_type == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "lr":
        return LogisticRegression(random_state=42)
    else:
        raise ValueError("Type de modèle non reconnu. Choisissez 'rf' ou 'lr'.")
