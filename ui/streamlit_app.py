import streamlit as st
import requests
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Product Review Risk Dashboard")

division = st.text_input("Division", "General")
department = st.text_input("Department", "Tops")
class_name = st.text_input("Class", "Blouses")
age = st.number_input("Typical reviewer age", value=35)
reviews_text = st.text_area("Paste reviews, one per line", height=200)

if st.button("Check risk"):
    reviews = [r.strip() for r in reviews_text.split("\n") if r.strip()]
    if not reviews:
        st.warning("Paste at least one review.")
    else:
        try:
            resp = requests.post(f"{API_URL}/predict", json={
                "reviews": reviews, "division": division,
                "department": department, "class_name": class_name, "age": age
            }, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            if result["status"] == "insufficient_data":
                st.info(f"Only {result['review_count']} reviews — need at least 3 for a reliable score.")
            else:
                st.metric("Risk score", result["risk_score"], result["risk_label"].upper())
                st.write("**Most concerning reviews:**")
                for r in result["most_negative_reviews"]:
                    st.write(f"- {r}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API: {e}")