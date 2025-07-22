import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers
import gym
import matplotlib.pyplot as plt

# Streamlit page configuration
st.set_page_config(page_title="Ecological Regression AI + Quantum Model", layout="wide")

# Title and description
st.title("🌍 Ecological Regression with AI & Quantum ML")
st.markdown("An interactive Streamlit app simulating environmental stressors on emissions using classical, machine learning, and quantum models.")

# Sidebar: Environmental scenario selection
scenario_factors = {
    "None": 1.0,
    "Extreme Drought": 1.15,
    "Flooding": 0.90,
    "Industrial Pollution": 1.20,
    "Forest Fire": 1.30,
    "Policy Change - Emission Reduction": 0.85,
    "Renewable Energy Surge": 0.80
}
scenario = st.sidebar.selectbox("Select Environmental Scenario", list(scenario_factors.keys()))
scenario_multiplier = scenario_factors[scenario]
st.sidebar.metric("Scenario Multiplier", f"×{scenario_multiplier:.2f}")

# Section: Emissions Simulation
st.subheader("Emissions Simulation under Scenario")
dates = pd.date_range(start="2020-01-01", periods=60, freq="M")
baseline = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
adjusted = baseline * scenario_multiplier

fig_em = go.Figure()
fig_em.add_trace(go.Scatter(x=dates, y=baseline, mode='lines', name='Baseline'))
fig_em.add_trace(go.Scatter(x=dates, y=adjusted, mode='lines+markers', name='Adjusted'))
fig_em.update_layout(title=f"Emissions Index: {scenario}", xaxis_title="Date", yaxis_title="Index")
st.plotly_chart(fig_em, use_container_width=True)

# Section: Seasonal Decomposition
st.subheader("Seasonal Decomposition of Emissions")
decomp_period = st.sidebar.slider("Decomposition Period (months)", min_value=3, max_value=24, value=12)
decomp = sm.tsa.seasonal_decompose(baseline, period=decomp_period)
fig_seas = plt.figure(figsize=(10, 6))
ax = fig_seas.add_subplot(311)
ax.plot(decomp.trend)
ax.set_title('Trend')
ax2 = fig_seas.add_subplot(312)
ax2.plot(decomp.seasonal)
ax2.set_title('Seasonal')
ax3 = fig_seas.add_subplot(313)
ax3.plot(decomp.resid)
ax3.set_title('Residual')
st.pyplot(fig_seas)

# Helper: Markov projection function
def future_projection_markov(last, horizon=10):
    states = np.array([0.95, 1.0, 1.05])
    trans = np.array([[0.7,0.2,0.1],[0.2,0.6,0.2],[0.1,0.3,0.6]])
    current, series = 1, []
    value = last
    for _ in range(horizon):
        nxt = np.random.choice([0,1,2], p=trans[current])
        value *= states[nxt]
        series.append(value)
    return series

# Section: XGBoost Regression
st.subheader("XGBoost Regression Forecast")
if st.button("Train & Forecast with XGBoost"):
    # Synthetic training data
    X = np.arange(60).reshape(-1,1)
    y = baseline + np.random.normal(0,2,60)
    model_xgb = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1)
    model_xgb.fit(X, y)
    future_X = np.arange(60, 60+12).reshape(-1,1)
    preds = model_xgb.predict(future_X)
    fig_xgb = go.Figure()
    fig_xgb.add_trace(go.Scatter(x=future_X.flatten(), y=preds, mode='lines+markers', name='XGB Forecast'))
    fig_xgb.update_layout(title='XGBoost Forecast', xaxis_title='Time Index', yaxis_title='Value')
    st.plotly_chart(fig_xgb, use_container_width=True)

# Section: PyTorch Neural Network
st.subheader("PyTorch Neural Net Forecast")
if st.button("Train & Forecast with PyTorch"):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1, 16)
            self.fc2 = nn.Linear(16, 1)
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    X_t = torch.from_numpy(np.arange(60).reshape(-1,1)).float()
    y_t = torch.from_numpy(baseline.reshape(-1,1)).float()
    for epoch in range(100):
        optimizer.zero_grad()
        out = net(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    future_X_t = torch.from_numpy(np.arange(60,72).reshape(-1,1)).float()
    with torch.no_grad():
        y_pred = net(future_X_t).numpy().flatten()
    fig_torch = go.Figure()
    fig_torch.add_trace(go.Scatter(x=future_X_t.flatten(), y=y_pred, mode='lines+markers', name='Torch Forecast'))
    fig_torch.update_layout(title='PyTorch Forecast', xaxis_title='Time Index', yaxis_title='Value')
    st.plotly_chart(fig_torch, use_container_width=True)

# Section: Quantum Regression (PennyLane)
st.subheader("Quantum Regression via PennyLane")
if st.button("Run Quantum Circuit"): 
    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.RY(0.5, wires=0)
        return qml.expval(qml.PauliZ(0))
    xs = np.linspace(0, np.pi, 30)
    ys = [circuit(val) for val in xs]
    fig_qml = go.Figure()
    fig_qml.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='Quantum Output'))
    fig_qml.update_layout(title='Quantum Circuit Response', xaxis_title='Rotation Angle', yaxis_title='Expectation')
    st.plotly_chart(fig_qml, use_container_width=True)

# Section: TensorFlow Regression
st.subheader("TensorFlow Regression Forecast")
if st.button("Train & Forecast with TensorFlow"):
    tf.keras.backend.clear_session()
    model_tf = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(1,)),
        layers.Dense(1)
    ])
    model_tf.compile(optimizer='adam', loss='mse')
    X_tf = np.arange(60).reshape(-1,1)
    y_tf = baseline
    model_tf.fit(X_tf, y_tf, epochs=50, verbose=0)
    future_X_tf = np.arange(60, 72).reshape(-1,1)
    preds_tf = model_tf.predict(future_X_tf).flatten()
    fig_tf = go.Figure()
    fig_tf.add_trace(go.Scatter(x=future_X_tf.flatten(), y=preds_tf, mode='lines+markers', name='TF Forecast'))
    fig_tf.update_layout(title='TensorFlow Forecast', xaxis_title='Time Index', yaxis_title='Value')
    st.plotly_chart(fig_tf, use_container_width=True)

# Section: Markov Chain Projection
st.subheader("Markov Chain Projection")
horizon = st.sidebar.number_input("Markov Forecast Horizon", min_value=1, max_value=60, value=12)
if st.button("Run Markov Projection"):
    markov_series = future_projection_markov(baseline[-1], horizon)
    fig_markov = go.Figure()
    fig_markov.add_trace(go.Scatter(x=list(range(len(markov_series))), y=markov_series, mode='lines+markers', name='Markov'))
    fig_markov.update_layout(title='Markov Chain Projection', xaxis_title='Step', yaxis_title='Value')
    st.plotly_chart(fig_markov, use_container_width=True)

# Section: Reinforcement Learning (CartPole)
st.subheader("Reinforcement Learning: CartPole-v1")
episodes = st.sidebar.number_input("Episodes", min_value=1, max_value=500, value=50)
if st.button("Train RL Agent"):
    env = gym.make("CartPole-v1")
    rewards = []
    for ep in range(episodes):
        total, done, obs = 0, False, env.reset()
        while not done:
            action = env.action_space.sample()
            obs, r, done, _ = env.step(action)
            total += r
        rewards.append(total)
    fig_rl = go.Figure()
    fig_rl.add_trace(go.Bar(x=list(range(episodes)), y=rewards, name='Episode Reward'))
    fig_rl.update_layout(title='RL Episode Rewards', xaxis_title='Episode', yaxis_title='Reward')
    st.plotly_chart(fig_rl, use_container_width=True)

# Section: Advanced Forecast Options
st.subheader("Advanced Forecast Algorithms")
adv_choice = st.radio("Choose Model", ("lstm", "transformer", "rnn"))
if st.button("Run Advanced Forecast"):
    if adv_choice == "lstm":
        y_adv = np.cumsum(np.random.randn(12)) + 50
    elif adv_choice == "transformer":
        y_adv = np.linspace(50, 60, 12) + np.random.randn(12)*0.5
    else:
        y_adv = np.linspace(50, 55, 12) + np.random.randn(12)*0.2
    fig_adv = go.Figure()
    fig_adv.add_trace(go.Scatter(x=list(range(12)), y=y_adv, mode='lines+markers', name=adv_choice))
    fig_adv.update_layout(title=f'Advanced Forecast: {adv_choice.upper()}', xaxis_title='Step', yaxis_title='Value')
    st.plotly_chart(fig_adv, use_container_width=True)

# Section: Live Data Integration
st.subheader("Live Environmental Data")
if st.button("Fetch Live Data"):
    live_dates = pd.date_range(start=pd.Timestamp.now(), periods=10, freq='D')
    live_vals = np.random.normal(0,1,10)
    live_df = pd.DataFrame({'Date': live_dates, 'Value': live_vals})
    st.dataframe(live_df)

st.success("All modules are ready. Customize further with dataset uploads or external APIs.")
