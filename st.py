import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular
import scikitplot as skplt
from imblearn.over_sampling import RandomOverSampler
import openai
st.set_page_config(layout="wide")
import time

# Introduction statement
intro_statement = """
## ART Discontinuation Predictor

Hello there! Have you ever been concerned about the factors that lead to the discontinuation of Antiretroviral Therapy (ART) and wished you had the data to understand these patterns? This interactive application, utilizing data-driven predictive models, allows you to explore and predict the likelihood of ART discontinuation based on a variety of patient and treatment characteristics. Whether you're a healthcare provider aiming to enhance patient adherence or a researcher investigating treatment dynamics, this tool offers valuable insights to inform your decisions. For an optimal viewing experience, especially on mobile devices, we recommend switching to landscape mode. Dive in and uncover the critical factors influencing ART discontinuation!
"""
import streamlit as st

# Load the GIF
gif_path = 'C:\\Users\\Amell\\Downloads\\istockphoto-1208393826-612x612.jpg'

# Display the GIF with reduced size
st.image(gif_path, caption=None, use_column_width=True, width=200)

# Add CSS styling for hover effects
st.markdown(
    """
    <style>
        /* Hover effects */
        img:hover {
            transform: scale(1.05);  /* Scale up image by 5% on hover */
            transition: transform 0.3s ease-in-out;  /* Smooth transition */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add CSS styling for rounded edges
st.markdown(
    """
    <style>
        img {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Split the introduction statement into lines
lines = intro_statement.strip().split("\n")

# Display each line of the introduction statement with a delay
for line in lines:
    st.markdown(line)
    time.sleep(2)  # Adjust the sleep duration to control the speed

# Define the custom CSS for fonts
custom_css = """
/* Custom fonts */
h1 {
    font-family: 'Arial Black', sans-serif;
}

h2 {
    font-family: 'Georgia', serif;
}

h3 {
    font-family: 'Courier New', monospace;
}

p {
    font-family: 'Times New Roman', serif;
}
"""

# Apply the custom CSS
st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)
# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #1c1e21;
        color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #333;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ff6347;
    }
    .stButton button {
        background-color: #4b0082;
        color: white;
    }
    .stMetric {
        font-size: 1.5rem;
        color: #ff6347;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .card {
        background-color: #444;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .card h4 {
        margin-bottom: 15px;
    }
    .card p {
        color: #ccc;
    }
    </style>
    """, unsafe_allow_html=True
)

# Specify the correct file path
file_path = 'C:\\Users\\Amell\\Downloads\\streamlit\\dff.csv'
# Custom Transformer to convert boolean columns to integers
class BooleanConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        boolean_columns = X.select_dtypes(include='bool').columns
        X[boolean_columns] = X[boolean_columns].astype(int)
        return X

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Define the mapping for risk categories
    risk_mapping = {
        'In Treatment': 'Low',
        'IIT': 'High',
        'Transfer Out': 'Medium'
    }

    # Transform the 'Status' column based on the risk mapping
    df['Status'] = df['Status'].map(risk_mapping)

    # Drop rows where 'Status' is null (e.g., 'Died')
    df = df[df['Status'].notnull()]

    # Drop irrelevant columns
    df.drop(columns=['Patient_uid', 'EducationLevel', 'Region', 'SiteCode', 'LastVisit', 'VisitDate', 'NextAppointmentDate', 'NextVisit'], inplace=True)

    # Convert boolean columns to integers
    df = BooleanConverter().fit_transform(df)

    # Encode the target variable
    label_encoder = LabelEncoder()
    df['Status'] = label_encoder.fit_transform(df['Status'])

    # Select columns for one-hot encoding
    columns_to_encode = ['Gender', 'Occupation', 'MaritalStatus', 'StartRegimen', 'LastRegimen', 'ArtAdherence', 'PHQ_9_rating']

    # Perform one-hot encoding with boolean dtype
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
    df_encoded = df_encoded.astype(int)

    # Define the desired number of samples for each class
    sampling_strategy = {1: 17547, 0: 12000, 2: 15009}

    # Instantiate the RandomOverSampler with the specified sampling strategy
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)

    # Resample the dataset
    X = df_encoded.drop(columns=['Status'])
    y = df_encoded['Status']
    X_resampled, y_resampled = ros.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['Status'] = y_resampled

    return df, df_resampled, label_encoder
# Function to train the model
def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, accuracy, cm, report,X_test,X_train,y_test,y_pred
from sklearn.ensemble import RandomForestClassifier
# Load and preprocess the data
df, df_resampled, label_encoder = load_and_preprocess_data(file_path)

# Select features and target variable
target_col = 'Status'
X = df_resampled.drop(columns=[target_col])
y = df_resampled[target_col]
# Train the model
model, accuracy, cm, report,X_test,X_train,y_test,y_pred = train_model(X, y)
# Add custom CSS for the sidebar
cool_sidebar_css = """
.sidebar .sidebar-content {
    background-color: #272822;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    animation: slide-in-left 0.5s ease;
}

@keyframes slide-in-left {
    from {
        transform: translateX(-100%);
    }
    to {
        transform: translateX(0);
    }
}

.sidebar .sidebar-content .stFileUploader {
    margin-top: 20px;
}
"""
from bokeh.layouts import gridplot
from sklearn.metrics import precision_recall_curve,f1_score
# Assuming df is your DataFrame and y_test, y_pred, rf_classifier, and X_train are defined elsewhere
from bokeh.models import ColumnDataSource
from sklearn.preprocessing import label_binarize
import numpy as np



# Apply custom styling using st.markdown
st.markdown(f'<style>{cool_sidebar_css}</style>', unsafe_allow_html=True)

# Create a sidebar for uploading CSV file
st.sidebar.header("Upload CSV for Prediction")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    st.sidebar.write("File uploaded successfully!")

    # Load and preprocess the uploaded data
    df_uploaded = pd.read_csv(uploaded_file)
    df_processed, label_encoder = preprocess_data(df_uploaded)

    # Train the model
    model = train_model(df_processed.drop(columns=['Status']), df_processed['Status'])

    # Make predictions using the trained model
    predictions = make_predictions(model, df_processed)

    # Display the predictions
    st.write("Predictions:")
    st.write(predictions)

# Add Text Input widgets to the sidebar
search_query_1 = st.sidebar.text_input("comment", "  ")
import plotly.express as px    
st.title('🩺 Patient Risk Prediction and Explanation ')
st.markdown("Welcome to the Patient Risk Prediction Dashboard. Use the tabs to explore data, performance metrics, and predictions.")
import lime
import lime.lime_tabular
# Function to generate visualizations based on selected patient record
def generate_visualizations(gender):
    patient_data = df[df['Gender'] == gender]
    # Example visualization (you can add more as needed)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Risk_Category', data=patient_data, palette='viridis')
    plt.title('Risk Category Distribution for Patients with Gender {}'.format(gender))
    plt.xlabel('Risk Category')
    plt.ylabel('Count')
    st.pyplot()
# Define your main layout
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to", ["Data", "Performance", "Local Performance", "Chat with GPT", "Model Prediction"])
from sklearn.preprocessing import label_binarize
if selected_tab == "Data":
    st.header("📊 Data")
    st.write(df)

    if st.checkbox('Show Summary Statistics'):
        st.header("Summary Statistics")
        st.write(df.describe())

    if st.checkbox('Show Plots'):
        st.header("Interactive Plot")
        # Example: Interactive scatter plot using Plotly
        x_axis = st.selectbox("Select X-axis", options=df.columns)
        y_axis = st.selectbox("Select Y-axis", options=df.columns)
        fig = px.scatter(df, x=x_axis, y=y_axis)
        st.plotly_chart(fig)
    # Sidebar filters
    st.sidebar.subheader("Filters")

    # Filter by gender
    gender = st.sidebar.selectbox("Select Gender", df["Gender"].unique())

    # Filter by occupation
    occupation = st.sidebar.selectbox("Select Occupation", df["Occupation"].unique())

    # Filter by marital status
    marital_status = st.sidebar.selectbox("Select Marital Status", df["MaritalStatus"].unique())

    # Filter by ART adherence
    art_adherence = st.sidebar.selectbox("Select ART Adherence", df["ArtAdherence"].unique())

    # Apply filters
    filtered_df = df[(df["Gender"] == gender) & (df["Occupation"] == occupation) & 
                     (df["MaritalStatus"] == marital_status) & (df["ArtAdherence"] == art_adherence)]

    # Drop rows with non-numeric values in "PHQ_9_rating" column
    filtered_df = filtered_df.dropna(subset=["PHQ_9_rating"], axis=0)

    # Print unique values in "PHQ_9_rating" column
    st.write("Unique values in 'PHQ_9_rating' column:", filtered_df["PHQ_9_rating"].unique())

    # Display filtered data
    st.subheader("Filtered Data")
    st.write(filtered_df)

    # Aggregates and totals
    st.subheader("Aggregates and Totals")

    # Total count button
    if st.button("Total count", key="total_count_button", help="Click to view the total count of filtered data"):
               st.write("Total count:", len(filtered_df))

    # Median AgeLastVisit button
    if st.button("Median AgeLastVisit", key="median_age_button", help="Click to view the median age of last visit"):
          st.write("Median AgeLastVisit:", filtered_df["AgeLastVisit"].median())


    # Plot data
    st.subheader("Interactive Plots")

    # Bar plot
    fig1 = px.bar(filtered_df, x="MaritalStatus", y="AgeLastVisit", color="Gender", barmode="stack",
                  title="Age at Last Visit by Marital Status (Stacked Bar Plot)")
    fig1.update_traces(marker_color='green')  # Set marker color to green
    fig1.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))

    # Histogram
    fig2 = px.histogram(filtered_df, x="AgeARTStart", color="Gender", marginal="box",
                          title="Distribution of Age at ART Start (Histogram with Violin Plot)")
    fig2.update_traces(marker_color='yellow')  # Set marker color to yellow
    fig2.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))

    # Violin plot
    fig3 = px.violin(filtered_df, y="PHQ_9_rating", x="MaritalStatus", color="Gender",
                     title="Distribution of PHQ_9 Rating by Marital Status (Violin Plot)")
    fig3.update_traces(marker_color='blue')  # Set marker color to blue
    fig3.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))

    # Line plot
    fig4 = px.line(filtered_df, x="AgeLastVisit", y="PHQ_9_rating", color="Gender",
                    title="PHQ_9 Rating Over Time (Line Plot)")
    fig4.update_traces(line=dict(color='orange'))  # Set line color to orange
    fig4.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))

    # Display plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
    
    
    with col2:
        st.plotly_chart(fig3)
        st.plotly_chart(fig4)
    
elif selected_tab == "Performance":
    st.header("Confusion Matrix | Global Feature Importance | Recall and F1 Score Curves")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        conf_matrix_fig, ax = plt.subplots(figsize=(8, 6))
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax=ax)
        st.pyplot(conf_matrix_fig)
    
    # Feature Importances
    with col2:
        st.subheader("Feature Importances")
        feat_importance_fig, ax = plt.subplots(figsize=(8, 6))
        skplt.estimators.plot_feature_importances(model, feature_names=X_train.columns, ax=ax,
                                                  title="Random Forest Feature Importances",  x_tick_rotation=90)
        st.pyplot(feat_importance_fig)
    
    # Recall and F1 Score Curves
    st.subheader("Recall and F1 Score Curves")
    recall_curve_fig = plt.figure(figsize=(8, 6))
    y_prob = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=np.unique(y_test))
    precision = dict()
    recall = dict()
    f1 = dict()
    for i in range(len(np.unique(y_test))):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        plt.plot(recall[i], f1[i], marker='.', label=f"Class {i}")
    plt.xlabel('Recall')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    st.pyplot(recall_curve_fig)

    st.divider()
    st.header("Classification Report")
    st.code(classification_report(y_test,y_pred))

elif selected_tab == "Local Performance":
    st.header("Predict Risk for a New Patient")
    col1, col2 = st.columns(2, gap="medium")
    sliders = []
    
    with col1:
        st.markdown("### Input Features")
        for feature in X_train.columns:
            min_val = float(X_train[feature].min())
            max_val = float(X_train[feature].max())
            if min_val == max_val:
                min_val = 0
                max_val = 1
            feature_slider = st.slider(label=feature, min_value=min_val, max_value=max_val, value=min_val, help=f"Adjust the value for {feature}")
            sliders.append(feature_slider)

    with col2:
        st.markdown("### Model Prediction")
        with st.spinner('Predicting...'):
            prediction = model.predict([sliders])

        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.markdown("#### Prediction: <strong style='color:tomato;'>{}</strong>".format(prediction[0]), unsafe_allow_html=True)

        probs = model.predict_proba([sliders])
        probability = probs[0][prediction[0]]

        with col2:
            st.metric(label="Model Confidence", value="{:.2f}%".format(100 * probability), delta="{:.2f}%".format(100 * (probability - 0.5)))


elif selected_tab == "Chat with GPT":
    st.header("💬 Chat with GPT")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def add_message(role, content):
        st.session_state.messages.append({"role": role, "content": content})

    def get_response(prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5",
            messages=st.session_state.messages
        )
        return response.choices[0].message["content"]

    user_input = st.text_input("You:", key="user_input")

    if st.button("Send"):
        if user_input:
            add_message("user", user_input)
            response = get_response(user_input)
            add_message("assistant", response)

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**ChatGPT:** {message['content']}")


elif selected_tab == "Model Prediction":
    st.header("Model Prediction")
    
    import shap
    from tabulate import tabulate

    sampled_X_test = X_test.sample(n=15, random_state=42)
    sampled_y_test = y_test[sampled_X_test.index]

    # Create a SHAP explainer object
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the sampled test data
    shap_values = explainer.shap_values(sampled_X_test)

    # Define function to generate explanations for the sampled data
    def generate_explanations_for_sample():
        explanations = []
        for index in range(len(sampled_X_test)):
            prediction = model.predict(sampled_X_test.iloc[index].values.reshape(1, -1))[0]
            explanation = []  # Define explanation list within the loop
            for i, feature_name in enumerate(sampled_X_test.columns):
                feature_value = sampled_X_test.iloc[index, i]
                if feature_value != 0:
                    # Ensure prediction is within the valid range
                    if 0 <= prediction < shap_values.shape[2]:
                        contribution = shap_values[index, i, prediction]
                        explanation.append(f"{feature_name} (value: {feature_value}) contributed by {contribution:.3f}")
                    else:
                        explanation.append(f"{feature_name} (value: {feature_value})")
            explanations.append((index, prediction, explanation))
        return explanations

    # Define function to format explanations into tabular style
    def format_explanations(explanations, risk_category):
        formatted_explanations = []
        headers = ["Patient", "Prediction", "Explanation"]

        for index, prediction, explanation in explanations:
            formatted_explanation = ["Patient " + str(index + 1), label_encoder.inverse_transform([prediction])[0], "\n".join([f"- {exp}" for exp in explanation])]
            formatted_explanations.append(formatted_explanation)

        # Print the tabular style output
        title_color = ""
        if risk_category == "High":
            title_color = "red"  # Red color for High risk
        elif risk_category == "Medium":
            title_color = "yellow"  # Yellow color for Medium risk
        elif risk_category == "Low":
            title_color = "green"  # Green color for Low risk

        st.markdown(f"<h3 style='color: {title_color};'>{risk_category} Risk Patients Prediction and Explanations:</h3>", unsafe_allow_html=True)
        # Add CSS style to the table for alignment
        st.markdown("<style>table {border-collapse: collapse; width: 100%;} th, td {text-align: left; padding: 8px;} th {background-color: #f2f2f2;}</style>", unsafe_allow_html=True)
        st.table(formatted_explanations)

    # Filter explanations by risk category
    def filter_explanations_by_risk(all_explanations, risk_category):
        filtered_explanations = []
        for index, prediction, explanation in all_explanations:
            if label_encoder.inverse_transform([prediction])[0] == risk_category:
                filtered_explanations.append((index, prediction, explanation))
        return filtered_explanations

    # Generate explanations for sampled data
    all_explanations = generate_explanations_for_sample()

    # Iterating over each risk category and format explanations
    risk_categories = ['High', 'Medium', 'Low']
    for risk_category in risk_categories:
        filtered_explanations = filter_explanations_by_risk(all_explanations, risk_category)
        format_explanations(filtered_explanations, risk_category)

