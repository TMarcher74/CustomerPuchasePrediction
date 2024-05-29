import joblib
from sklearn.neighbors import KNeighborsClassifier

# Example data preparation
# Note: Replace this with your actual data loading code
evidence = [
    [0, 0.0, 0, 0.0, 1, 0.0, 0.2, 0.2, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0.0, 0, 0.0, 2, 64.0, 0.0, 0.1, 0.0, 0.0, 1, 2, 2, 1, 2, 1, 0]
]
labels = [0, 1]  # Example labels (0: No purchase, 1: Purchase)

def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

# Train the model
model = train_model(evidence, labels)

# Save the model
joblib.dump(model, 'shopping_model.pkl')
