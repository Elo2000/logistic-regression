
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns

# Read the data from the files
x_data = pd.read_csv('ex2_x_data.csv')
y_data = pd.read_csv('ex2_y_data.csv')

# Shuffle and mix the data
data = pd.concat([y_data, x_data], axis=1).sample(frac=1, random_state=42)

# Set the input features (X) and output labels (Y)
X = data.iloc[:, 1:].values
Y = data.iloc[:, 0].values

# Create a subset for classes 0-9
subset_indices = (Y >= 0) & (Y <= 9)
X_subset = X[subset_indices]
Y_subset = Y[subset_indices]

# Split the subset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_subset, Y_subset, test_size=0.33, random_state=42)

# Create the logistic regression model and train it on the training data
model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)

# Predict the class labels for the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# Calculate the probabilities for each class
y_pred_prob = model.predict_proba(X_test)

# Calculate the ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
    roc_auc[i] = roc_auc_score(y_test == i, y_pred_prob[:, i])

# Plot the ROC curve for each class
plt.figure()
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, color in zip(range(10), colors):
    plt.plot(fpr[i], tpr[i], color=color, label='Class %d (AUC = %0.2f)' % (i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()