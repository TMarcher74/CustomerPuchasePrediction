import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Constants
TEST_SIZE = 0.4

# Load data function
def load_data(filename):
    evidence = []
    labels = []

    month_index = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
                   'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}

    with open(filename) as f:
        reader = csv.DictReader(f)

        for row in reader:
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                month_index[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0,
            ])

            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return evidence, labels

# Train model function
def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

# Evaluate function
def evaluate(labels, predictions):
    sensitivity = float(sum((1 for label, prediction in zip(labels, predictions) if label == 1 and label == prediction))) / sum((1 for label in labels if label == 1))
    specificity = float(sum((1 for label, prediction in zip(labels, predictions) if label == 0 and label == prediction))) / sum((1 for label in labels if label == 0))
    return sensitivity, specificity

# Load data
filename = 'activity_data.csv'
evidence, labels = load_data(filename)

# Split data
X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=TEST_SIZE)

# Train model
model = train_model(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation metrics
sensitivity, specificity = evaluate(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions, output_dict=True)

# Convert classification report to DataFrame for easy visualization
report_df = pd.DataFrame(class_report).transpose()

# Initialize the Dash app
app = Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Online Shopping Behavior Analysis Dashboard"),

    # Accuracy and other metrics
    html.Div([
        html.H3(f"Accuracy: {accuracy * 100:.2f}%"),
        html.H3(f"Sensitivity (True Positive Rate): {sensitivity * 100:.2f}%"),
        html.H3(f"Specificity (True Negative Rate): {specificity * 100:.2f}%")
    ]),

    # Confusion matrix heatmap
    html.Div([
        dcc.Graph(
            id='confusion-matrix',
            figure=px.imshow(
                conf_matrix,
                labels=dict(x="Predicted", y="Actual"),
                x=['No Revenue', 'Revenue'],
                y=['No Revenue', 'Revenue'],
                text_auto=True,
                title="Confusion Matrix"
            )
        )
    ]),

    # Classification report
    html.Div([
        dcc.Graph(
            id='classification-report',
            figure=px.bar(
                report_df,
                x=report_df.index,
                y=['precision', 'recall', 'f1-score'],
                barmode='group',
                title="Classification Report"
            )
        )
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
