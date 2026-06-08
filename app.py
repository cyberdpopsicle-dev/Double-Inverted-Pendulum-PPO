import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from stable_baselines3 import PPO
from lime import lime_tabular

from double_pendulum_env import DoubleInvertedPendulumEnv


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Robotic Double Pendulum ML Dashboard",
    layout="wide"
)

st.title(" Interactive ML Capstone Control Center")
st.markdown(
    "### Wheel-Based Stair Climbing Device (SCD) Edge-Pivot Controller Explainability Platform"
)


# ============================================================
# ENVIRONMENT
# ============================================================

env = DoubleInvertedPendulumEnv()


# ============================================================
# LOAD PPO MODEL
# ============================================================

@st.cache_resource
def load_trained_agent():
    try:
        model = PPO.load("ppo_double_pendulum_model")
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


trained_model = load_trained_agent()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def fallback_controller(state):
    error_1 = np.pi - state[0]
    error_2 = np.pi - state[2]

    torque = (
        (error_1 * 3.5)
        + (error_2 * 1.5)
        - (state[1] * 0.4)
    )

    return np.clip(torque, -6.0, 6.0)


def predict_torque(X_input):

    X_input = np.asarray(X_input)

    if trained_model is not None:

        if X_input.ndim == 1:
            action, _ = trained_model.predict(
                X_input,
                deterministic=True
            )

            return np.array([float(np.squeeze(action))])

        outputs = []

        for row in X_input:
            action, _ = trained_model.predict(
                row,
                deterministic=True
            )

            outputs.append(float(np.squeeze(action)))

        return np.array(outputs)

    else:

        if X_input.ndim == 1:
            return np.array([fallback_controller(X_input)])

        outputs = []

        for row in X_input:
            outputs.append(fallback_controller(row))

        return np.array(outputs)


def predict_torque_lime(X_inputs):
    return predict_torque(X_inputs)


# ============================================================
# USER INPUTS
# ============================================================

col_input, col_output = st.columns([1, 2])

with col_input:

    st.header(" Manual State Inputs")

    phi1 = st.slider(
        "Chassis Link Angle φ1",
        -float(np.pi),
        float(np.pi),
        float(np.pi),
        0.01
    )

    phi1_dot = st.slider(
        "Chassis Angular Velocity φ1_dot",
        -5.0,
        5.0,
        0.0,
        0.1
    )

    phi2 = st.slider(
        "Wheel Link Angle φ2",
        -float(np.pi),
        float(np.pi),
        float(np.pi),
        0.01
    )

    phi2_dot = st.slider(
        "Wheel Angular Velocity φ2_dot",
        -5.0,
        5.0,
        0.0,
        0.1
    )

    current_state = np.array(
        [phi1, phi1_dot, phi2, phi2_dot],
        dtype=np.float32
    )

    feature_names = [
        "phi1",
        "phi1_dot",
        "phi2",
        "phi2_dot"
    ]

    st.info(f"Current State: {np.round(current_state,3)}")

    st.header(" XAI Interpretability Engine")

    xai_method = st.radio(
        "Explanation Method",
        [
            "SHAP",
            "LIME"
        ]
    )


# ============================================================
# MODEL STATUS
# ============================================================

if trained_model is not None:

    action, _ = trained_model.predict(
        current_state,
        deterministic=True
    )

    st.success(
        f" PPO model loaded successfully | Test Action = {float(np.squeeze(action)):.4f}"
    )

else:

    st.warning(
        " PPO model not found. Using fallback controller."
    )


# ============================================================
# PREDICTION
# ============================================================

predicted_u = float(
    predict_torque(current_state)[0]
)


# ============================================================
# SHAP BACKGROUND
# ============================================================

np.random.seed(42)

background_data = np.random.uniform(
    -np.pi,
    np.pi,
    size=(100, 4)
)


@st.cache_resource
def get_shap_explainer():

    return shap.KernelExplainer(
        predict_torque,
        background_data
    )


# ============================================================
# OUTPUT PANEL
# ============================================================

with col_output:

    st.header(" Real-Time Evaluation & Telemetry")

    st.metric(
        "Predicted Torque",
        f"{predicted_u:.4f} Nm",
        delta="Limit ±6 Nm"
    )

    # --------------------------------------------------------
    # FORWARD SIMULATION
    # --------------------------------------------------------

    sim_states = []

    temp_state = current_state.copy()

    for _ in range(50):

        u = float(
            predict_torque(temp_state)[0]
        )

        env.state = temp_state.copy()

        step_result = env.step(
            np.array([u], dtype=np.float32)
        )

        if len(step_result) == 5:

            obs, reward, terminated, truncated, info = step_result

            temp_state = obs

            if terminated or truncated:
                break

        else:

            temp_state = step_result

        sim_states.append(
            temp_state.copy()
        )

    sim_df = pd.DataFrame(
        sim_states,
        columns=feature_names
    )

    sim_df["Time (ms)"] = (
        np.arange(len(sim_df)) * 2
    )

    # --------------------------------------------------------
    # TELEMETRY CHART
    # --------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sim_df["Time (ms)"],
            y=sim_df["phi1"],
            mode="lines",
            name="φ1"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sim_df["Time (ms)"],
            y=sim_df["phi2"],
            mode="lines",
            name="φ2"
        )
    )

    fig.update_layout(
        title="100 ms Forward Trajectory Prediction",
        xaxis_title="Time (ms)",
        yaxis_title="Angle (rad)",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.subheader(
        "Feature Importance Analysis"
    )

    if xai_method == "SHAP":

        explainer = get_shap_explainer()

        shap_values = explainer.shap_values(
            current_state,
            nsamples=100
        )

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_values
        })

        shap_df = shap_df.sort_values(
            by="Impact"
        )

        fig_shap = go.Figure()

        fig_shap.add_trace(
            go.Bar(
                x=shap_df["Impact"],
                y=shap_df["Feature"],
                orientation="h"
            )
        )

        fig_shap.update_layout(
            title="SHAP Local Feature Contributions",
            height=300
        )

        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )

    else:

        explainer = lime_tabular.LimeTabularExplainer(
            background_data,
            feature_names=feature_names,
            mode="regression"
        )

        exp = explainer.explain_instance(
            current_state,
            predict_torque_lime,
            num_features=4
        )

        weights = exp.as_list()

        features = [x[0] for x in weights]
        values = [x[1] for x in weights]

        fig_lime = go.Figure()

        fig_lime.add_trace(
            go.Bar(
                x=values,
                y=features,
                orientation="h"
            )
        )

        fig_lime.update_layout(
            title="LIME Local Explanation",
            height=300
        )

        st.plotly_chart(
            fig_lime,
            use_container_width=True
        )
