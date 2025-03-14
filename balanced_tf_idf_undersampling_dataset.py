import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler

# Load preprocessed dataset
df = pd.read_csv("preprocessed_socmed_dataset.csv")

# Handle NaN values in 'Cleaned_Captions'
df['Cleaned_Captions'] = df['Cleaned_Captions'].fillna('')  # Replace NaNs with empty strings

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
X_tfidf = tfidf_vectorizer.fit_transform(df['Cleaned_Captions'])

# Convert to DataFrame
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Target variable
y = df['Label']

# Apply Undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_tfidf_df, y)

# Combine features and target into a single DataFrame using pd.concat
balanced_df = pd.concat([pd.DataFrame(X_resampled, columns=tfidf_vectorizer.get_feature_names_out()), 
                         pd.Series(y_resampled, name='Label')], axis=1)

# Save the balanced dataset
balanced_filename = "balanced_tf_idf_undersampling_dataset.csv"
balanced_df.to_csv(balanced_filename, index=False)

print(f"✅ TF-IDF and Undersampling completed. Balanced dataset saved as '{balanced_filename}'.")

