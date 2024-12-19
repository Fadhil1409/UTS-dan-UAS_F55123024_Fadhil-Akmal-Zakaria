import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load the dataset
def load_dataset(file_path):
    # Load dataset
    dataset = pd.read_csv(file_path, delimiter=';')

    # Clean dataset: Drop irrelevant columns and handle numeric conversion
    dataset = dataset.drop(columns=["Unnamed: 15", "Unnamed: 16"], errors='ignore')
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            dataset[col] = dataset[col].str.replace(',', '.').astype('float', errors='ignore')

    # Drop rows with missing values
    dataset = dataset.dropna()

    # Add a new column for air quality classification based on CO(GT)
    dataset['AirQuality'] = ['Good' if co < 2.5 else 'Bad' for co in dataset['CO(GT)']]

    return dataset

# 2. Prepare data (train-test split and encoding labels)
def prepare_data(df):
    features = ['CO(GT)', 'PT08.S1(CO)']
    label = 'AirQuality'

    X = df[features]
    y = df[label]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le

# 3. Train the K-NN model
def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# 4. Evaluate the model
def evaluate_model(knn, X_test, y_test, label_encoder):
    y_pred = knn.predict(X_test)

    # Decode labels
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_decoded, y_pred_decoded))
    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))

# 5. Cross-validation and recording accuracies
def cross_validate_model(knn, X, y):
    cv_scores = cross_val_score(knn, X, y, cv=5)  # 5-fold cross-validation
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean()}")
    return cv_scores

# 6. Plot decision boundaries
def plot_decision_boundaries(knn, X, y):
    h = 0.1  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.xlabel('CO(GT)')
    plt.ylabel('PT08.S1(CO)')
    plt.title('K-NN Decision Boundary')
    plt.show()

# 7. Plot accuracy graph
def plot_accuracy_graph(train_accuracies, test_accuracies, val_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(test_accuracies, label='Testing Accuracy', marker='s')
    plt.plot(val_accuracies, label='Validation Accuracy', marker='^')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load dataset
    file_path = r"D:\UAS KB\Dataset\AirQuality.csv" # Update path to your dataset
    df = load_dataset(file_path)

    # Prepare data
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(df)

    # Train K-NN model
    knn_model = train_knn(X_train, y_train, n_neighbors=5)

    # Evaluate the model
    evaluate_model(knn_model, X_test, y_test, label_encoder)

    # Perform cross-validation and get accuracy scores
    cv_scores = cross_validate_model(knn_model, X_train, y_train)

    # Plot decision boundaries (using train data)
    plot_decision_boundaries(knn_model, X_train.values, y_train)
    plt.show() 

    # Plot accuracy graph (example data, replace with actual training/test accuracies)
    train_accuracies = [0.85, 0.87, 0.88, 0.89, 0.90]  # Replace with actual values
    test_accuracies = [0.80, 0.82, 0.83, 0.85, 0.86]   # Replace with actual values
    val_accuracies = [0.81, 0.83, 0.84, 0.85, 0.86]    # Replace with actual values
    plot_accuracy_graph(train_accuracies, test_accuracies, val_accuracies)
    
    y_pred_knn = knn(X_train, y_train, X_test)
    print("KNN Classification Report:")
    print(classification_report(y_test, y_pred_knn))
    display_confusion_matrix(y_test, y_pred_knn, "KNN Confusion Matrix")
