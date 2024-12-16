import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pysindy as ps
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from scipy.interpolate import UnivariateSpline, BSpline
from scipy.special import gamma
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import catboost as cb
except ImportError:
    cb = None

try:
    from pysr import PySRRegressor
except ImportError:
    PySRRegressor = None

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None

from sklearn.datasets import load_diabetes


###############################
# Data Generation Functions
###############################

def generate_dynamic_sine_data(num_samples=1000, T=1.0, num_timesteps=50, noise_std=0.01):
    dt = T / num_timesteps
    theta = 1.0
    sigma = 0.1
    base_freq = 2.0 * np.pi
    X_data = []
    Y_data = []
    for _ in range(num_samples):
        z = 0.0
        z_path = []
        for i in range(num_timesteps):
            dz = -theta * z * dt + sigma * np.sqrt(dt) * np.random.randn()
            z += dz
            z_path.append(z)
        z_path = np.array(z_path)
        t_grid = np.linspace(0, T, num_timesteps)
        omega = base_freq + z_path
        y = np.sin(omega * t_grid) + noise_std * np.random.randn(num_timesteps)
        X_data.append(z_path[:, None])
        Y_data.append(y)
    return np.array(X_data), np.array(Y_data), np.linspace(0, 1, num_timesteps)


def generate_dynamic_beta_data(num_samples=1000, T=1.0, num_timesteps=50, noise_std=0.01):
    dt = T / num_timesteps
    theta = 1.0
    sigma = 0.1
    X_data = []
    Y_data = []
    for _ in range(num_samples):
        z = 0.0
        z_path = []
        for i in range(num_timesteps):
            dz = -theta * z * dt + sigma * np.sqrt(dt) * np.random.randn()
            z += dz
            z_path.append(z)
        z_path = np.array(z_path)
        t_grid = np.linspace(1e-3, 1.0 - 1e-3, num_timesteps)
        alpha_vals = 2 + z_path
        beta_vals = 2 + 0.5 * z_path
        y = []
        for i, (a, b) in enumerate(zip(alpha_vals, beta_vals)):
            val = (t_grid[i] ** (a - 1) * (1 - t_grid[i]) ** (b - 1)) / (gamma(a) * gamma(b) / gamma(a + b))
            val += noise_std * np.random.randn()
            y.append(val)
        y = np.array(y)
        X_data.append(z_path[:, None])
        Y_data.append(y)
    return np.array(X_data), np.array(Y_data), np.linspace(0, 1, num_timesteps)


def load_sp500_data(start='2019-01-01', end='2020-01-01', time_steps=50, step=1):
    if pdr is None:
        raise ImportError("pandas_datareader is not installed. Please install it.")
    df = pdr.DataReader('SP500', 'fred', start, end)
    df = df.dropna()
    close = df['SP500'].values
    N = len(close)
    if N <= time_steps:
        raise ValueError("Not enough data to form samples for SP500 dataset.")

    close_norm = (close - close.mean()) / close.std()
    returns = np.diff(np.log(close + 1))
    returns = np.concatenate(([0], returns))
    returns = (returns - returns.mean()) / returns.std()

    samples_X = []
    samples_Y = []
    for start_idx in range(0, N - time_steps, step):
        end_idx = start_idx + time_steps
        if end_idx > N:
            break
        x_segment = returns[start_idx:end_idx][:, None]
        y_segment = close_norm[start_idx:end_idx]
        if len(x_segment) == time_steps:
            samples_X.append(x_segment)
            samples_Y.append(y_segment)
    samples_X = np.array(samples_X)
    samples_Y = np.array(samples_Y)
    t_grid = np.linspace(0, 1, time_steps)
    return samples_X, samples_Y, t_grid


def load_elec_data(filepath='LD2011_2014.txt', time_steps=50, step=1):
    df = pd.read_csv(filepath, sep=';', index_col=0, decimal=',')
    df = df.dropna(axis=1)
    data = df.iloc[:, 0].values
    data_norm = (data - data.mean()) / data.std()

    N = len(data_norm)
    if N <= time_steps:
        raise ValueError("Not enough data to form samples.")

    offset = abs(data_norm.min()) + 1 if data_norm.min() < 0 else 1
    returns = np.diff(np.log(data_norm + offset))
    returns = np.concatenate(([0], returns))

    samples_X = []
    samples_Y = []
    for start in range(0, N - time_steps, step):
        end = start + time_steps
        if end > N:
            break
        x_segment = returns[start:end][:, None]
        y_segment = data_norm[start:end]
        if len(x_segment) == time_steps:
            samples_X.append(x_segment)
            samples_Y.append(y_segment)

    samples_X = np.array(samples_X)
    samples_Y = np.array(samples_Y)
    t_grid = np.linspace(0, 1, time_steps)
    return samples_X, samples_Y, t_grid


def load_diabetes_data(time_steps=10, step=1):
    d = load_diabetes()
    X = d.data
    Y = d.target
    Y_norm = (Y - Y.mean()) / Y.std()

    N = len(Y)
    if N <= time_steps:
        raise ValueError("Not enough data to form sequences")

    feature_0 = X[:, 0]
    f0 = (feature_0 - feature_0.mean()) / feature_0.std()

    samples_X = []
    samples_Y = []
    for start_idx in range(0, N - time_steps, step):
        end_idx = start_idx + time_steps
        if end_idx > N:
            break
        x_seg = f0[start_idx:end_idx][:, None]
        y_seg = Y_norm[start_idx:end_idx]
        if len(x_seg) == time_steps:
            samples_X.append(x_seg)
            samples_Y.append(y_seg)

    samples_X = np.array(samples_X)
    samples_Y = np.array(samples_Y)
    t_grid = np.linspace(0, 1, time_steps)
    return samples_X, samples_Y, t_grid


###############################
# Utility Functions
###############################

def split_data(X, Y, ratios=(0.7, 0.15, 0.15)):
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    tr_end = int(ratios[0] * n)
    val_end = int((ratios[0] + ratios[1]) * n)
    return (X[idx[:tr_end]], Y[idx[:tr_end]]), (X[idx[tr_end:val_end]], Y[idx[tr_end:val_end]]), (X[idx[val_end:]], Y[idx[val_end:]])


def add_time_feature(X):
    N, T, M = X.shape
    t_grid = np.linspace(0, 1, T)
    t_col = np.tile(t_grid, (N, 1))
    X_new = np.concatenate([X, t_col[:, :, None]], axis=2)
    return X_new


def flatten_data(X, Y):
    N, T, M = X.shape
    Xf = X.reshape(N * T, M)
    Yf = Y.reshape(N * T)
    return Xf, Yf


def prepare_symbolic_data(X, Y):
    N, T, M = X.shape
    t_grid = np.linspace(0, 1, T)
    XX = []
    YY = []
    for i in range(N):
        for j in range(T):
            row = np.concatenate(([t_grid[j]], X[i, j, :]))
            XX.append(row)
            YY.append(Y[i, j])
    return np.array(XX), np.array(YY)


def select_knots(X_time, Y_data, desired_knots=5):
    candidate_knots = []
    for t_arr, y_arr in zip(X_time, Y_data):
        s_low = 0.0
        s_high = 1e5
        found = False
        for _ in range(50):
            s_mid = (s_low + s_high) / 2
            sp = UnivariateSpline(t_arr, y_arr, s=s_mid, k=3)
            kts = sp.get_knots()
            if len(kts) - 2 > desired_knots:
                s_high = s_mid
            elif len(kts) - 2 < desired_knots:
                s_low = s_mid
            else:
                candidate_knots.extend(kts[1:-1])
                found = True
                break
        if not found:
            sp = UnivariateSpline(t_arr, y_arr, s=s_mid, k=3)
            kts = sp.get_knots()
            candidate_knots.extend(kts[1:-1])
    if len(candidate_knots) < desired_knots:
        return np.linspace(0, 1, desired_knots)
    candidate_knots = np.array(candidate_knots).reshape(-1, 1)
    kmeans = KMeans(n_clusters=desired_knots, random_state=42).fit(candidate_knots)
    knots = sorted(kmeans.cluster_centers_.flatten())
    return np.array(knots)


def bspline_basis(t_grid, knots, degree=3):
    t0 = 0.0
    t1 = 1.0
    t_aug = np.concatenate(([t0] * degree, knots, [t1] * degree))
    B = len(t_aug) - degree - 1
    basis_matrix = np.zeros((len(t_grid), B))
    for i in range(B):
        c = np.zeros(B)
        c[i] = 1
        spline = BSpline(t_aug, c, degree)
        basis_matrix[:, i] = spline(t_grid)
    return basis_matrix, t_aug


###############################
# Models
###############################

class DeltaTRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim + 1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        N, T, M = x.shape
        dt = (1.0 / T) * torch.ones(N, T, 1)
        inp = torch.cat([x, dt], dim=2)
        h0 = torch.zeros(1, x.size(0), self.hidden_dim)
        out, _ = self.rnn(inp, h0)
        yhat = self.fc(out)
        return yhat.squeeze(-1)


def train_torch_model(model, trainX, trainY, valX, valY, testX, testY, epochs=20):
    device = torch.device("cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    Xtr = torch.tensor(trainX, dtype=torch.float32)
    Ytr = torch.tensor(trainY, dtype=torch.float32)
    Xv = torch.tensor(valX, dtype=torch.float32)
    Yv = torch.tensor(valY, dtype=torch.float32)
    Xte = torch.tensor(testX, dtype=torch.float32)
    Yte = torch.tensor(testY, dtype=torch.float32)

    best_val = np.inf
    best_model = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        yhat = model(Xtr)
        loss = criterion(yhat, Ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            yhv = model(Xv)
            vloss = criterion(yhv, Yv).item()
        if vloss < best_val:
            best_val = vloss
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        yhte = model(Xte)
        test_mse = criterion(yhte, Yte).item()
    return test_mse, model


def train_tree_model(model, trainX, trainY, valX, valY, testX, testY):
    trainXf, trainYf = flatten_data(trainX, trainY)
    testXf, testYf = flatten_data(testX, testY)
    model.fit(trainXf, trainYf)
    pred = model.predict(testXf)
    mse = mean_squared_error(testYf, pred)
    return mse


def train_linear(trainX, trainY, valX, valY, testX, testY):
    trainXf, trainYf = flatten_data(trainX, trainY)
    testXf, testYf = flatten_data(testX, testY)
    reg = LinearRegression()
    reg.fit(trainXf, trainYf)
    pred = reg.predict(testXf)
    mse = mean_squared_error(testYf, pred)
    return mse


def run_pysr(trainX, trainY, testX, testY):
    if PySRRegressor is None:
        return None
    XX, YY = prepare_symbolic_data(trainX, trainY)
    XX_test, YY_test = prepare_symbolic_data(testX, testY)
    model = PySRRegressor(
        niterations=200,
        unary_operators=["sin", "cos", "exp", "log"],
        binary_operators=["+", "-", "*", "/"],
        model_selection="best",
        maxsize=10,
        progress=False,
        random_state=42
    )
    model.fit(XX, YY)
    pred_test = model.predict(XX_test)
    mse = mean_squared_error(YY_test, pred_test)
    return mse


def run_sindy(trainX, trainY, testX, testY):
    XX, YY = prepare_symbolic_data(trainX, trainY)
    N, T, M = trainX.shape
    t_grid = XX[:T, 0]
    x_list = []
    for i in range(N):
        start = i * T
        end = (i + 1) * T
        XX_i = XX[start:end, :]
        YY_i = YY[start:end]
        traj_i = np.hstack([YY_i[:, None], XX_i[:, 2:]])
        x_list.append(traj_i)
    model_sindy = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3))
    model_sindy.fit(x_list, t=t_grid, multiple_trajectories=True)
    XX_test, YY_test = prepare_symbolic_data(testX[:1], testY[:1])
    YYt = YY_test[:T]
    test_traj = np.hstack([YYt[:, None], XX_test[:T, 2:]])
    init_state = test_traj[0]
    y_sim = model_sindy.simulate(init_state, t_grid)
    y_sim_y = y_sim[:, 0]
    mse = mean_squared_error(YYt, y_sim_y)
    return mse


def train_decision_tree(trainX, trainY, valX, valY, testX, testY):
    dt_reg = DecisionTreeRegressor(max_depth=5)
    return train_tree_model(dt_reg, trainX, trainY, valX, valY, testX, testY)


class DYCOTE_RNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, B):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, B)

    def forward(self, x, basis_matrix):
        out, _ = self.rnn(x)
        c = self.fc(out)  # (N,T,B)
        c_reshape = c.unsqueeze(3)
        basis_reshape = torch.tensor(basis_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(3)
        yhat = (c_reshape * basis_reshape).sum(dim=2).squeeze(-1)
        return yhat


def train_dycote(model, trainX, trainY, valX, valY, testX, testY, basis_matrix, epochs=100, lr=0.001):
    device = torch.device("cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    Xtr = torch.tensor(trainX, dtype=torch.float32)
    Ytr = torch.tensor(trainY, dtype=torch.float32)
    Xv = torch.tensor(valX, dtype=torch.float32)
    Yv = torch.tensor(valY, dtype=torch.float32)
    Xte = torch.tensor(testX, dtype=torch.float32)
    Yte = torch.tensor(testY, dtype=torch.float32)

    best_val = np.inf
    best_model = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        yhat = model(Xtr, basis_matrix)
        loss = criterion(yhat, Ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            yhv = model(Xv, basis_matrix)
            vloss = criterion(yhv, Yv).item()
        if vloss < best_val:
            best_val = vloss
            best_model = model.state_dict()
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        yhte = model(Xte, basis_matrix)
        test_mse = criterion(yhte, Yte).item()
    return test_mse, model


if odeint is not None:
    class NeuralODEFunc(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.act = nn.Tanh()
            self.out = nn.Linear(hidden_dim, input_dim)

        def forward(self, t, z):
            h = self.act(self.fc(z))
            dz = self.out(h)
            return dz

    class DYCOTE_NeuralODE_Model(nn.Module):
        def __init__(self, input_dim, hidden_dim, B, latent_dim=4):
            super().__init__()
            self.latent_dim = latent_dim
            self.encoder = nn.Linear(input_dim, latent_dim)
            self.func = NeuralODEFunc(latent_dim, hidden_dim)
            self.fc = nn.Linear(latent_dim, B)

        def forward(self, x, basis_matrix):
            N, T, M = x.shape
            device = x.device
            z0 = self.encoder(x[:, 0, :])
            t_grid = torch.linspace(0, 1, T, device=device)
            z_traj = odeint(self.func, z0, t_grid, method='rk4')  # (T,N,latent_dim)
            z_traj = z_traj.permute(1, 0, 2)  # (N,T,latent_dim)
            coeff = self.fc(z_traj)  # (N,T,B)
            c_reshape = coeff.unsqueeze(3)
            basis_reshape = torch.tensor(basis_matrix, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(3)
            yhat = (c_reshape * basis_reshape).sum(dim=2).squeeze(-1)
            return yhat

    def train_dycote_ode(model, trainX, trainY, valX, valY, testX, testY, basis_matrix, epochs=100, lr=0.001):
        device = torch.device("cpu")
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        Xtr = torch.tensor(trainX, dtype=torch.float32)
        Ytr = torch.tensor(trainY, dtype=torch.float32)
        Xv = torch.tensor(valX, dtype=torch.float32)
        Yv = torch.tensor(valY, dtype=torch.float32)
        Xte = torch.tensor(testX, dtype=torch.float32)
        Yte = torch.tensor(testY, dtype=torch.float32)

        best_val = np.inf
        best_model = None
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            yhat = model(Xtr, basis_matrix)
            loss = criterion(yhat, Ytr)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                yhv = model(Xv, basis_matrix)
                vloss = criterion(yhv, Yv).item()
            if vloss < best_val:
                best_val = vloss
                best_model = model.state_dict()
        model.load_state_dict(best_model)
        model.eval()
        with torch.no_grad():
            yhte = model(Xte, basis_matrix)
            test_mse = criterion(yhte, Yte).item()
        return test_mse, model
else:
    train_dycote_ode = None


###############################
# Additional Utility Functions
###############################

def extract_motifs_from_y(y):
    dy = np.gradient(y)
    d2y = np.gradient(dy)

    def classify(sf, ss):
        s1 = np.sign(sf)
        s2 = np.sign(ss)
        if s1 > 0 and s2 > 0:
            return "s++"
        elif s1 > 0 and s2 < 0:
            return "s+-"
        elif s1 < 0 and s2 > 0:
            return "s-+"
        elif s1 < 0 and s2 < 0:
            return "s--"
        elif s1 > 0 and s2 == 0:
            return "s+0"
        elif s1 < 0 and s2 == 0:
            return "s-0"
        elif s1 == 0 and s2 == 0:
            return "s00"
        return "??"

    curr = classify(dy[0], d2y[0])
    start_idx = 0
    motifs = []
    for i in range(1, len(y)):
        m = classify(dy[i - 1], d2y[i - 1])
        if m != curr:
            motifs.append((start_idx, i, curr))
            start_idx = i
            curr = m
    motifs.append((start_idx, len(y) - 1, curr))
    return motifs


def stability_test(model, Xsample, basis_matrix, epsilon=0.01):
    X_pert = Xsample.copy()
    X_pert[:, :, 0] += epsilon * np.random.randn(*X_pert[:, :, 0].shape)
    with torch.no_grad():
        Ybase = model(torch.tensor(Xsample, dtype=torch.float32), basis_matrix).numpy()[0]
        Ypert = model(torch.tensor(X_pert, dtype=torch.float32), basis_matrix).numpy()[0]
    motifs_base = extract_motifs_from_y(Ybase)
    motifs_pert = extract_motifs_from_y(Ypert)
    return motifs_base, motifs_pert


def vary_covariate(model, Xsample, basis_matrix, idx=0, var_range=(-0.1, 0.1), steps=5):
    with torch.no_grad():
        base_pred = model(torch.tensor(Xsample, dtype=torch.float32), basis_matrix).numpy()[0]
    results = []
    shifts = np.linspace(var_range[0], var_range[1], steps)
    for v in shifts:
        X_mod = Xsample.copy()
        X_mod[:, :, idx] += v
        with torch.no_grad():
            pred = model(torch.tensor(X_mod, dtype=torch.float32), basis_matrix).numpy()[0]
        motifs_pred = extract_motifs_from_y(pred)
        results.append((v, pred, motifs_pred))
    return base_pred, results


###############################
# Pipeline
###############################

def run_pipeline(dataset_type='sine'):
    # Generate/Load data
    if dataset_type == 'sine':
        X_data, Y_data, t_grid = generate_dynamic_sine_data()
    elif dataset_type == 'beta':
        X_data, Y_data, t_grid = generate_dynamic_beta_data()
    elif dataset_type == 'sp500':
        X_data, Y_data, t_grid = load_sp500_data(start='2019-01-01', end='2020-01-01', time_steps=50, step=1)
        if len(X_data) < 100:
            print("Not enough data samples from SP500 for this configuration.")
            return
    elif dataset_type == 'elec':
        X_data, Y_data, t_grid = load_elec_data(filepath='LD2011_2014.txt', time_steps=50, step=1)
        if len(X_data) < 100:
            print("Not enough data from Electricity dataset for this configuration.")
            return
    elif dataset_type == 'diabetes':
        X_data, Y_data, t_grid = load_diabetes_data(time_steps=10, step=1)
        if len(X_data) < 100:
            print("Not enough diabetes data for this configuration.")
            return
    else:
        raise ValueError("dataset_type must be 'sine', 'beta', 'sp500', 'elec', or 'diabetes'")

    (trainX, trainY), (valX, valY), (testX, testY) = split_data(X_data, Y_data)
    trainX_t = add_time_feature(trainX)
    valX_t = add_time_feature(valX)
    testX_t = add_time_feature(testX)

    desired_knots = 5
    X_time = [t_grid for _ in range(trainX_t.shape[0])]
    knots = select_knots(X_time, trainY, desired_knots)
    basis_matrix, t_aug = bspline_basis(t_grid, knots, degree=3)
    B = basis_matrix.shape[1]
    print("Number of basis functions:", B)

    # ∆t-RNN
    delta_rnn_model = DeltaTRNN(input_dim=trainX.shape[2], hidden_dim=32, output_dim=1)
    delta_rnn_mse, _ = train_torch_model(delta_rnn_model, trainX, trainY, valX, valY, testX, testY)

    # XGB-T
    xgb_mse = None
    if xgboost_is_installed := (xgb is not None):
        xgb_model = xgb.XGBRegressor(n_estimators=100, verbosity=0)
        xgb_mse = train_tree_model(xgb_model, trainX, trainY, valX, valY, testX, testY)

    # LGBM-T
    lgb_mse = None
    if lgb_is_installed := (lgb is not None):
        lgb_model = lgb.LGBMRegressor(n_estimators=100)
        lgb_mse = train_tree_model(lgb_model, trainX, trainY, valX, valY, testX, testY)

    # CatBoost-T
    cat_mse = None
    if cb_is_installed := (cb is not None):
        cat_model = cb.CatBoostRegressor(iterations=100, verbose=False)
        cat_mse = train_tree_model(cat_model, trainX, trainY, valX, valY, testX, testY)

    # PySR
    pysr_mse = None
    if PySRRegressor is not None:
        pysr_mse = run_pysr(trainX, trainY, testX, testY)

    # SINDy
    sindy_mse = None
    try:
        sindy_mse = run_sindy(trainX, trainY, testX, testY)
    except Exception:
        sindy_mse = None

    # Linear-T
    lin_mse = train_linear(trainX_t, trainY, valX_t, valY, testX_t, testY)

    # DecisionTree-T
    dt_mse = train_decision_tree(trainX_t, trainY, valX_t, valY, testX_t, testY)

    # DYCOTE-RNN
    dycote_rnn = DYCOTE_RNN_Model(input_dim=trainX_t.shape[2], hidden_dim=64, B=B)
    dycote_rnn_mse, dycote_rnn_trained = train_dycote(dycote_rnn, trainX_t, trainY, valX_t, valY, testX_t, testY, basis_matrix, epochs=1000, lr=0.0003)

    # DYCOTE-NeuralODE
    dycote_ode_mse = None
    if train_dycote_ode is not None:
        dycote_ode = DYCOTE_NeuralODE_Model(input_dim=trainX_t.shape[2], hidden_dim=32, B=B, latent_dim=4)
        dycote_ode_mse, dycote_ode_trained = train_dycote_ode(dycote_ode, trainX_t, trainY, valX_t, valY, testX_t, testY, basis_matrix, epochs=1000, lr=0.0003)

    print(f"MSE results (Dynamic {dataset_type.capitalize()} dataset):")
    print("∆t-RNN:", delta_rnn_mse)
    if xgb_is_installed:
        print("XGB-T:", xgb_mse)
    else:
        print("XGB-T: not installed")
    if lgb_is_installed:
        print("LGBM-T:", lgb_mse)
    else:
        print("LGBM-T: not installed")
    if cb_is_installed:
        print("CatBoost-T:", cat_mse)
    else:
        print("CatBoost-T: not installed")
    print("PySR:", pysr_mse)
    print("SINDy:", sindy_mse)
    print("Linear-T:", lin_mse)
    print("DecisionTree-T:", dt_mse)
    print("DYCOTE-RNN:", dycote_rnn_mse)
    print("DYCOTE-NeuralODE:", dycote_ode_mse)

    # Extract motifs for one test sample (DYCOTE-RNN)
    dycote_rnn_trained.eval()
    with torch.no_grad():
        y_pred = dycote_rnn_trained(torch.tensor(testX_t[0:1], dtype=torch.float32), basis_matrix).numpy().squeeze(0)
    motifs = extract_motifs_from_y(y_pred)
    print("Motifs for one test sample (DYCOTE-RNN):", motifs)
    print("Number of motifs:", len(motifs))

    # Plot trajectory
    plt.figure(figsize=(8, 4))
    plt.plot(t_grid, y_pred, label='DYCOTE-RNN prediction')
    plt.title(f'DYCOTE-RNN Predicted Trajectory (Dynamic {dataset_type.capitalize()})')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.savefig('trajectory_plot.png')  # Save plot instead of show for non-interactive use
    plt.close()

    # Evaluate stability
    motifs_base, motifs_pert = stability_test(dycote_rnn_trained, testX_t[0:1], basis_matrix, epsilon=0.01)
    print("Motifs before perturbation:", motifs_base)
    print("Motifs after perturbation:", motifs_pert)
    print("Motif count stability:", len(motifs_base), "->", len(motifs_pert))

    # Vary covariate
    base_pred, vary_results = vary_covariate(dycote_rnn_trained, testX_t[0:1], basis_matrix, idx=0, var_range=(-0.1, 0.1), steps=5)
    for v, pred, motifs_pred in vary_results:
        print(f"Shift {v:.2f} on covariate 0, new motifs: {motifs_pred}")

    # Plot how trajectory changes with covariate perturbation
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid, base_pred, label='Base')
    for (v, pred, motifs_pred) in vary_results:
        plt.plot(t_grid, pred, label=f'X0+={v:.2f}')
    plt.title('Trajectory changes with covariate perturbation (DYCOTE-RNN)')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.savefig('covariate_perturbation_plot.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end pipeline for a chosen dataset.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="sine",
        choices=["sine", "beta", "sp500", "elec", "diabetes"],
        help="Type of dataset to run the pipeline on.",
    )
    args = parser.parse_args()

    run_pipeline(dataset_type=args.dataset_type)


if __name__ == "__main__":
    main()
