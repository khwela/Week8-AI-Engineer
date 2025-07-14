import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

st.title('💉 AI Health Monitor System')
st.write('This health monitor app will detect if you have normal blood oxygen or heart rate')

# Load the pre-trained model
model = joblib.load('/workspaces/Week8-AI-Engineer/models/anomaly_model.pkl')

# Anomaly detection function
def detect_anomalies(new_data):
    prediction = model.predict([new_data])  # returns 1 (normal), -1 (anomaly)
    return 'Anomaly' if prediction[0] == -1 else 'Normal'

# Recommendation engine
def get_recommendation(status, heart_rate, oxygen):
    if status == 'Anomaly':
        if heart_rate > 100:
            return "⚠️ High heart rate detected. Please rest or consult a doctor."
        elif oxygen < 92:
            return "⚠️ Low blood oxygen. Consider breathing exercises or medical help."
        else:
            return "⚠️ Irregular pattern detected. Monitor your health."
    return "✅ All metrics are within a healthy range. Keep it up!"

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your wearable data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df)

    results = []
    for _, row in df.iterrows():
        status = detect_anomalies([row['heart_rate'], row['blood_oxygen']])
        recommendation = get_recommendation(status, row['heart_rate'], row['blood_oxygen'])
        
        # Append results for saving later
        results.append({
            'timestamp': row['timestamp'],
            'heart_rate': row['heart_rate'],
            'blood_oxygen': row['blood_oxygen'],
            'activity_level': row.get('activity_level', 'N/A'),
            'status': status,
            'recommendation': recommendation
        })

        # Display 
        st.markdown(f"**Time:** {row['timestamp']} — Status: `{status}`")
        st.markdown(f"*{recommendation}*")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Button to save results
    if st.button("💾 Save Results to CSV"):
        output_path = '/workspaces/Week8-AI-Engineer/models/result.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        st.success(f"✅ Results saved to {output_path}")

