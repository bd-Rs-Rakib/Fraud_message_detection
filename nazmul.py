import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import numpy as np

# Load or train model
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pydata-dc-2016-tutorial/master/sms.tsv"

    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

@st.cache_data
def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )
    
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    return tfidf, model, accuracy, confusion_matrix(y_test, y_pred)

# Streamlit app
st.title("ML-Powered Fraud Message Detector 🕵️♂️")
st.write("This app uses machine learning to detect potential fraudulent messages.")

# Sidebar controls
threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)

# Load model and vectorizer
tfidf, model, accuracy, cm = train_model()

# Store vectorizer and model in session state
if 'model' not in st.session_state:
    st.session_state.model = model
if 'tfidf' not in st.session_state:
    st.session_state.tfidf = tfidf

# Input section
user_input = st.text_area("Enter the message to analyze:", height=150)

if st.button("Analyze Message"):
    if not user_input.strip():
        st.warning("Please enter a message to analyze.")
    else:
        # Preprocess input
        X_input = st.session_state.tfidf.transform([user_input])
        
        # Get prediction probabilities
        proba = st.session_state.model.predict_proba(X_input)[0]
        fraud_prob = proba[1]
        
        # Make prediction based on threshold
        prediction = 1 if fraud_prob >= threshold else 0
        
        # Display results
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Fraud Probability", f"{fraud_prob*100:.1f}%")
            
        with col2:
            if prediction == 1:
                st.error("⚠️ Fraud Alert! This message appears suspicious")
            else:
                st.success("✅ This message appears legitimate")
        
        # Show probability distribution
        st.markdown("### Probability Distribution")
        prob_df = pd.DataFrame({
            'Category': ['Legitimate', 'Fraudulent'],
            'Probability': [proba[0], proba[1]]
        })
        st.bar_chart(prob_df.set_index('Category'))

# Model info section
with st.expander("Model Details"):
    st.write(f"**Accuracy:** {accuracy:.2%}")
    st.write("**Confusion Matrix:**")
    st.write(pd.DataFrame(cm, 
             columns=['Predicted Legit', 'Predicted Fraud'],
             index=['Actual Legit', 'Actual Fraud']))
    
    st.write("**Top Predictive Features:**")
    feature_names = tfidf.get_feature_names_out()
    coefficients = model.coef_[0]
    top_features = pd.DataFrame({
        'Feature': feature_names,
        'Weight': coefficients
    }).sort_values('Weight', ascending=False)
    st.dataframe(top_features.head(10))

st.markdown("---")
st.markdown("""
**Note:** 
- Model trained on SMS Spam Collection Dataset
- Threshold can be adjusted in the sidebar
- Actual fraud detection systems would use more sophisticated models and data
""")