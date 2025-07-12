import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')

# Convert categorical columns to numbers
df['Interest'] = df['Interest'].astype('category').cat.codes
df['Career_Path'] = df['Career_Path'].astype('category')
target_names = df['Career_Path'].cat.categories
df['Career_Path'] = df['Career_Path'].cat.codes

# Features and Target
X = df[['Age', 'Marks_Math', 'Marks_Eng', 'Interest']]
y = df['Career_Path']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and category labels
pickle.dump((model, target_names), open('model.pkl', 'wb'))
