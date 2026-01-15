

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from galacto_models import load_models


st.set_page_config(
    page_title="GALACTO 🚀",
    layout="centered"
)


st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0b1c2d, #000000);
    color: white;
}
.stButton>button {
    background-color: #1f77ff;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


models = load_models()

mission_model = models["mission_model"]
cost_model = models["cost_model"]
success_model = models["success_model"]
encoders = models["encoders"]
features = models["features"]


raw_df = pd.read_csv("space_missions_dataset.csv")

st.title("🌌 GALACTO")
st.subheader("AI Space Mission Planning System")

st.markdown("""
Predict:
- 🛰 **Mission Type**
- 💰 **Mission Cost**
- 📈 **Mission Success Probability**
""")


user_input = {}

for feature in features:
    if feature in encoders:
        user_input[feature] = st.selectbox(
            feature,
            encoders[feature].classes_
        )
    else:
        user_input[feature] = st.number_input(
            feature,
            min_value=0.0,
            step=1.0
        )


def plot_cost_comparison(target, predicted_cost):
    target_df = raw_df[raw_df["Target Type"] == target]

    if target_df.empty:
        st.warning("No historical missions available for this target.")
        return

    fig, ax = plt.subplots()

    ax.plot(
        target_df["Mission Name"],
        target_df["Mission Cost (billion USD)"],
        marker="o",
        linestyle="-",
        label="Past Missions"
    )

    ax.axhline(
        y=predicted_cost,
        color="red",
        linestyle="--",
        label="Your Estimated Cost"
    )

    ax.set_title(f"Historical Cost Comparison for {target} Missions")
    ax.set_ylabel("Cost (Billion USD)")
    ax.set_xlabel("Mission Name")
    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)


if st.button("🚀 Launch Prediction"):
    input_row = []

    for feature in features:
        if feature in encoders:
            value = encoders[feature].transform([user_input[feature]])[0]
        else:
            value = user_input[feature]
        input_row.append(value)

    input_array = np.array(input_row).reshape(1, -1)

    mission_pred = mission_model.predict(input_array)[0]
    mission_name = encoders["Mission Type"].inverse_transform([mission_pred])[0]

    cost_pred = cost_model.predict(input_array)[0]
    success_pred = success_model.predict(input_array)[0]

    st.success("🛰️ Mission Analysis Complete")

    st.markdown(f"""
    ### 🧠 Prediction Results
    - **Mission Type:** `{mission_name}`
    - **Estimated Cost:** `${cost_pred:.2f} Billion`
    - **Success Probability:** `{success_pred:.2f}%`
    """)

    st.subheader("📊 Cost Benchmarking with Past Missions")
    plot_cost_comparison(
        target=user_input["Target Type"],
        predicted_cost=cost_pred
    )


st.markdown("---")
st.markdown("👨‍🚀 *Made by Prem Ashtekar*")
