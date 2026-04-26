"""
ids_complete.py
===============
Complete Intrusion Detection System — all-in-one runner.

Runs:
  1. Data generation  (simulates NSL-KDD if real files absent)
  2. Preprocessing    (encode → map attacks → engineer features → scale)
  3. Isolation Forest training
  4. Random Forest training
  5. Dual-model evaluation
  6. Dash dashboard   (http://127.0.0.1:8050)

Usage:
    pip install scikit-learn pandas numpy dash plotly
    python ids_complete.py
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import os, pickle, warnings
warnings.filterwarnings("ignore")

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
import dash
from dash import dcc, html
import plotly.graph_objs as go

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────────────────
CATEGORY_NAMES = [
    "Normal", "Brute Force", "Data Exfiltration",
    "Geo Anomaly", "Privilege Escalation", "Insider Threat",
]

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack_type", "difficulty",
]

ATTACK_MAPPING = {
    "normal": 0,
    "guess_passwd": 1, "ftp_write": 1,
    "warezclient": 2, "warezmaster": 2, "spy": 2, "imap": 2,
    "portsweep": 3, "ipsweep": 3, "nmap": 3, "satan": 3,
    "buffer_overflow": 4, "loadmodule": 4, "rootkit": 4, "perl": 4,
    "multihop": 5, "phf": 5,
    "back": 3, "land": 3, "neptune": 3, "pod": 3, "smurf": 3, "teardrop": 3,
}

SELECTED_FEATURES = [
    "duration_seconds", "is_long_connection", "session_rate",
    "src_bytes_log", "dst_bytes_log", "transfer_ratio",
    "connection_count", "srv_connection_count", "connection_rate",
    "error_rate_total", "srv_error_rate_total",
    "is_logged_in", "failed_login_count",
]

# ────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA (load real files OR generate synthetic)
# ────────────────────────────────────────────────────────────────────────────
def _generate_synthetic(n_train: int = 12_000, n_test: int = 3_000) -> tuple:
    """
    Produce a realistic-looking NSL-KDD-shaped DataFrame when real files
    are absent. Each attack class gets its own statistical signature.
    """
    rng = np.random.default_rng(42)

    def _make_rows(n, attack_label):
        cat = ATTACK_MAPPING.get(attack_label, 0)

        # Tweak signal strength per category
        dur_mean   = [0,  30,  120, 2,   5,  200][cat]
        src_mean   = [1e3, 5e2, 5e5, 8e3, 2e4, 1e6][cat]
        err_mean   = [0.01, 0.3, 0.05, 0.5, 0.1, 0.05][cat]
        logged_p   = [0.6,  0.1, 0.8,  0.2, 0.9,  0.7][cat]
        fail_mean  = [0.1, 5.0, 0.2,  1.0, 2.0,  0.5][cat]

        rows = {
            "duration":         np.abs(rng.normal(dur_mean, dur_mean + 1, n)),
            "protocol_type":    rng.choice(["tcp", "udp", "icmp"], n),
            "service":          rng.choice(["http", "ftp", "smtp", "ssh", "other"], n),
            "flag":             rng.choice(["SF", "S0", "REJ", "RSTO"], n),
            "src_bytes":        np.abs(rng.normal(src_mean, src_mean * 0.5 + 1, n)),
            "dst_bytes":        np.abs(rng.normal(src_mean * 0.3, src_mean * 0.2 + 1, n)),
            "land":             rng.integers(0, 2, n),
            "wrong_fragment":   rng.integers(0, 3, n),
            "urgent":           rng.integers(0, 2, n),
            "hot":              rng.integers(0, 10, n),
            "num_failed_logins":np.abs(rng.poisson(fail_mean, n)).astype(int),
            "logged_in":        rng.choice([0, 1], n, p=[1 - logged_p, logged_p]),
            "num_compromised":  rng.integers(0, 5, n),
            "root_shell":       rng.integers(0, 2, n),
            "su_attempted":     rng.integers(0, 2, n),
            "num_root":         rng.integers(0, 5, n),
            "num_file_creations":rng.integers(0, 5, n),
            "num_shells":       rng.integers(0, 3, n),
            "num_access_files": rng.integers(0, 5, n),
            "num_outbound_cmds":np.zeros(n, int),
            "is_host_login":    rng.integers(0, 2, n),
            "is_guest_login":   rng.integers(0, 2, n),
            "count":            rng.integers(1, 512, n),
            "srv_count":        rng.integers(1, 512, n),
            "serror_rate":      np.clip(rng.normal(err_mean, 0.1, n), 0, 1),
            "srv_serror_rate":  np.clip(rng.normal(err_mean, 0.1, n), 0, 1),
            "rerror_rate":      np.clip(rng.normal(err_mean * 0.5, 0.05, n), 0, 1),
            "srv_rerror_rate":  np.clip(rng.normal(err_mean * 0.5, 0.05, n), 0, 1),
            "same_srv_rate":    np.clip(rng.normal(0.8, 0.2, n), 0, 1),
            "diff_srv_rate":    np.clip(rng.normal(0.2, 0.1, n), 0, 1),
            "srv_diff_host_rate":np.clip(rng.normal(0.1, 0.05, n), 0, 1),
            "dst_host_count":   rng.integers(1, 256, n),
            "dst_host_srv_count":rng.integers(1, 256, n),
            "dst_host_same_srv_rate":np.clip(rng.normal(0.7, 0.2, n), 0, 1),
            "dst_host_diff_srv_rate":np.clip(rng.normal(0.2, 0.1, n), 0, 1),
            "dst_host_same_src_port_rate":np.clip(rng.normal(0.5, 0.2, n), 0, 1),
            "dst_host_srv_diff_host_rate":np.clip(rng.normal(0.1, 0.05, n), 0, 1),
            "dst_host_serror_rate":np.clip(rng.normal(err_mean, 0.1, n), 0, 1),
            "dst_host_srv_serror_rate":np.clip(rng.normal(err_mean, 0.1, n), 0, 1),
            "dst_host_rerror_rate":np.clip(rng.normal(err_mean * 0.5, 0.05, n), 0, 1),
            "dst_host_srv_rerror_rate":np.clip(rng.normal(err_mean * 0.5, 0.05, n), 0, 1),
            "attack_type":      [attack_label] * n,
            "difficulty":       rng.integers(1, 21, n),
        }
        return pd.DataFrame(rows)

    def _build_split(total):
        # rough class distribution: 60% normal, rest attacks
        parts = [
            _make_rows(int(total * 0.60), "normal"),
            _make_rows(int(total * 0.08), "guess_passwd"),
            _make_rows(int(total * 0.04), "warezclient"),
            _make_rows(int(total * 0.14), "portsweep"),
            _make_rows(int(total * 0.04), "buffer_overflow"),
            _make_rows(int(total * 0.05), "multihop"),
            _make_rows(int(total * 0.05), "neptune"),
        ]
        df = pd.concat(parts, ignore_index=True)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("  (No real data files found — generating synthetic NSL-KDD-shaped data)")
    return _build_split(n_train), _build_split(n_test)


def load_data():
    print("\n" + "="*60)
    print("STEP 1 — LOADING DATA")
    print("="*60)

    train_path = "data/raw/KDDTrain+.txt"
    test_path  = "data/raw/KDDTest+.txt"

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("  Loading real NSL-KDD files …")
        train_df = pd.read_csv(train_path, names=COLUMN_NAMES, header=None)
        test_df  = pd.read_csv(test_path,  names=COLUMN_NAMES, header=None)
    else:
        train_df, test_df = _generate_synthetic()

    print(f"   Training samples : {len(train_df):,}")
    print(f"   Test samples     : {len(test_df):,}")
    return train_df, test_df


# ────────────────────────────────────────────────────────────────────────────
# STEP 2 — PREPROCESSING
# ────────────────────────────────────────────────────────────────────────────
def encode_categorical(df):
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def map_attack_types(df):
    df["attack_category"] = df["attack_type"].map(ATTACK_MAPPING).fillna(3).astype(int)
    return df


def engineer_features(df):
    df["duration_seconds"]    = df["duration"]
    df["is_long_connection"]  = (df["duration"] > 60).astype(int)
    df["session_rate"]        = df["count"] / (df["duration"] + 1)
    df["src_bytes_log"]       = np.log1p(df["src_bytes"])
    df["dst_bytes_log"]       = np.log1p(df["dst_bytes"])
    df["transfer_ratio"]      = df["src_bytes"] / (df["dst_bytes"] + 1)
    df["connection_count"]    = df["count"]
    df["srv_connection_count"]= df["srv_count"]
    df["connection_rate"]     = df["count"] / (df["duration"] + 1)
    df["error_rate_total"]    = df["serror_rate"] + df["rerror_rate"]
    df["srv_error_rate_total"]= df["srv_serror_rate"] + df["srv_rerror_rate"]
    df["is_logged_in"]        = df["logged_in"]
    df["failed_login_count"]  = df["num_failed_logins"]
    return df


def preprocess_pipeline(train_df, test_df):
    print("\n" + "="*60)
    print("STEP 2 — PREPROCESSING")
    print("="*60)

    for df in (train_df, test_df):
        encode_categorical(df)
        map_attack_types(df)
        engineer_features(df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[SELECTED_FEATURES].values)
    y_train = train_df["attack_category"].values
    X_test  = scaler.transform(test_df[SELECTED_FEATURES].values)
    y_test  = test_df["attack_category"].values

    print(f"   X_train shape : {X_train.shape}")
    print(f"   X_test  shape : {X_test.shape}")

    print("\n  Class distribution (training):")
    for i, name in enumerate(CATEGORY_NAMES):
        cnt = (y_train == i).sum()
        print(f"    {i}: {name:22s} — {cnt:6,d} ({cnt/len(y_train)*100:5.1f}%)")

    return X_train, y_train, X_test, y_test, scaler


# ────────────────────────────────────────────────────────────────────────────
# STEP 3 — ISOLATION FOREST
# ────────────────────────────────────────────────────────────────────────────
def train_isolation_forest(X_train, y_train):
    print("\n" + "="*60)
    print("STEP 3 — ISOLATION FOREST TRAINING")
    print("="*60)

    X_normal = X_train[y_train == 0]
    print(f"  Training on {len(X_normal):,} normal samples …")

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_normal)
    print("  ✓ Isolation Forest trained!")
    return iso


def evaluate_isolation_forest(iso, X_test, y_test):
    y_pred_iso    = iso.predict(X_test)
    y_pred_binary = np.where(y_pred_iso == -1, 1, 0)
    y_test_binary = np.where(y_test == 0, 0, 1)

    acc  = accuracy_score(y_test_binary, y_pred_binary)
    prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    rec  = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    f1   = f1_score(y_test_binary, y_pred_binary, zero_division=0)

    print(f"\n  Isolation Forest — Accuracy:{acc*100:.2f}%  "
          f"Precision:{prec*100:.2f}%  Recall:{rec*100:.2f}%  F1:{f1*100:.2f}%")
    return acc, prec, rec, f1


# ────────────────────────────────────────────────────────────────────────────
# STEP 4 — RANDOM FOREST
# ────────────────────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train):
    print("\n" + "="*60)
    print("STEP 4 — RANDOM FOREST TRAINING")
    print("="*60)
    print("  Training on all samples with balanced class weights …")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    rf.fit(X_train, y_train)
    print("   Random Forest trained!")
    return rf


def evaluate_random_forest(rf, X_test, y_test):
    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Random Forest — Accuracy:{acc*100:.2f}%  "
          f"Precision:{prec*100:.2f}%  Recall:{rec*100:.2f}%  F1:{f1*100:.2f}%")
    print("\n  Detailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=CATEGORY_NAMES,
                                zero_division=0, digits=4))

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    print("  Top-5 Features:")
    for i in range(min(5, len(SELECTED_FEATURES))):
        print(f"    {i+1}. {SELECTED_FEATURES[idx[i]]:25s} {importances[idx[i]]:.4f}")

    return acc, prec, rec, f1


# ────────────────────────────────────────────────────────────────────────────
# STEP 5 — DUAL-MODEL PIPELINE
# ────────────────────────────────────────────────────────────────────────────
def dual_model_predict(X, iso, rf):
    iso_pred   = iso.predict(X)
    final_pred = np.zeros(len(X), dtype=int)
    anomaly_idx = np.where(iso_pred == -1)[0]
    if len(anomaly_idx):
        final_pred[anomaly_idx] = rf.predict(X[anomaly_idx])
    return final_pred, iso_pred


def evaluate_dual_model(iso, rf, X_test, y_test):
    print("\n" + "="*60)
    print("STEP 5 — DUAL-MODEL PIPELINE EVALUATION")
    print("="*60)

    y_pred, iso_pred = dual_model_predict(X_test, iso, rf)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred, labels=list(range(6)))

    print(f"\n  Overall Accuracy : {acc*100:.2f}%")

    attack_mask    = y_test != 0
    total_attacks  = attack_mask.sum()
    flagged        = ((iso_pred == -1) & attack_mask).sum()
    correct_cls    = ((y_pred == y_test) & attack_mask).sum()

    print(f"\n  Stage-1 (Isolation Forest): {flagged:,}/{total_attacks:,} attacks flagged "
          f"({flagged/max(total_attacks,1)*100:.1f}%)")
    print(f"  Stage-2 (Random Forest)   : {correct_cls:,}/{total_attacks:,} correctly classified "
          f"({correct_cls/max(total_attacks,1)*100:.1f}%)")

    print("\n  Per-attack-type detection rate:")
    for i in range(1, 6):
        mask = y_test == i
        if mask.sum():
            det  = ((y_pred == i) & mask).sum()
            tot  = mask.sum()
            print(f"    {CATEGORY_NAMES[i]:25s}: {det:4d}/{tot:4d} ({det/tot*100:5.1f}%)")

    total_evts    = len(y_test)
    det_attacks   = int(((y_pred == y_test) & attack_mask).sum())
    det_rate      = det_attacks / max(total_attacks, 1) * 100

    return {
        "accuracy":       acc,
        "cm":             cm,
        "y_pred":         y_pred,
        "y_test":         y_test,
        "iso_pred":       iso_pred,
        "total_events":   total_evts,
        "total_attacks":  int(total_attacks),
        "det_attacks":    det_attacks,
        "det_rate":       det_rate,
    }


# ────────────────────────────────────────────────────────────────────────────
# STEP 6 — DASH DASHBOARD
# ────────────────────────────────────────────────────────────────────────────
COLORS = ["#22c55e", "#ef4444", "#f97316", "#eab308", "#a855f7", "#dc2626"]

def build_dashboard(results: dict, rf: RandomForestClassifier):
    acc         = results["accuracy"]
    cm          = results["cm"]
    y_pred      = results["y_pred"]
    y_test      = results["y_test"]
    total_evts  = results["total_events"]
    det_attacks = results["det_attacks"]
    det_rate    = results["det_rate"]

    attack_counts = [int((y_pred == i).sum()) for i in range(6)]

    per_class_rate = []
    for i in range(1, 6):
        mask = y_test == i
        if mask.sum():
            per_class_rate.append(
                round(((y_pred == i) & mask).sum() / mask.sum() * 100, 2)
            )
        else:
            per_class_rate.append(0.0)

    importances = rf.feature_importances_
    feat_sorted = sorted(zip(SELECTED_FEATURES, importances),
                         key=lambda x: x[1], reverse=True)
    feat_names  = [f[0] for f in feat_sorted]
    feat_vals   = [round(f[1], 4) for f in feat_sorted]

    app = dash.Dash(__name__)

    # ── Layout ──────────────────────────────────────────────────────────────
    app.layout = html.Div(style={
        "fontFamily": "'Courier New', monospace",
        "background": "#0a0e1a",
        "minHeight": "100vh",
        "padding": "24px",
        "color": "#e2e8f0",
    }, children=[

        # ── Header ──────────────────────────────────────────────────────────
        html.Div(style={
            "borderBottom": "1px solid #1e3a5f",
            "marginBottom": "28px",
            "paddingBottom": "16px",
        }, children=[
            html.Div("[ CLOUD IDS — DUAL MODEL PIPELINE ]", style={
                "fontSize": "22px", "fontWeight": "bold",
                "color": "#38bdf8", "letterSpacing": "3px",
            }),
            html.Div("Isolation Forest  ›  Random Forest  ·  NSL-KDD Dataset", style={
                "fontSize": "12px", "color": "#64748b", "marginTop": "4px",
                "letterSpacing": "2px",
            }),
        ]),

        # ── KPI row ─────────────────────────────────────────────────────────
        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "24px"},
        children=[
            _kpi("TOTAL EVENTS",    f"{total_evts:,}",      "#38bdf8"),
            _kpi("ATTACKS DETECTED",f"{det_attacks:,}",     "#ef4444"),
            _kpi("DETECTION RATE",  f"{det_rate:.1f}%",     "#22c55e"),
            _kpi("OVERALL ACCURACY",f"{acc*100:.1f}%",      "#a855f7"),
        ]),

        # ── Row 1: pie + confusion matrix ───────────────────────────────────
        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
        children=[
            html.Div(style=_card(flex=True), children=[
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Pie(
                            labels=CATEGORY_NAMES,
                            values=attack_counts,
                            hole=0.45,
                            marker=dict(colors=COLORS,
                                        line=dict(color="#0a0e1a", width=2)),
                            textfont=dict(size=11),
                        )],
                        layout=go.Layout(
                            title=dict(text="Predicted Attack Distribution",
                                       font=dict(color="#94a3b8", size=13)),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#cbd5e1"),
                            legend=dict(font=dict(size=10)),
                            margin=dict(t=50, b=10, l=10, r=10),
                        ),
                    ),
                    config={"displayModeBar": False},
                    style={"height": "340px"},
                )
            ]),

            html.Div(style=_card(flex=True), children=[
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Heatmap(
                            z=cm.tolist(),
                            x=CATEGORY_NAMES,
                            y=CATEGORY_NAMES,
                            colorscale="Blues",
                            text=cm.tolist(),
                            texttemplate="%{text}",
                            textfont=dict(size=9),
                            showscale=True,
                        )],
                        layout=go.Layout(
                            title=dict(text="Confusion Matrix",
                                       font=dict(color="#94a3b8", size=13)),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#cbd5e1", size=10),
                            xaxis=dict(title="Predicted", tickangle=-25,
                                       tickfont=dict(size=9)),
                            yaxis=dict(title="Actual", tickfont=dict(size=9)),
                            margin=dict(t=50, b=80, l=120, r=20),
                        ),
                    ),
                    config={"displayModeBar": False},
                    style={"height": "340px"},
                )
            ]),
        ]),

        # ── Row 2: per-class detection + feature importance ──────────────────
        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
        children=[
            html.Div(style=_card(flex=True), children=[
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Bar(
                            x=CATEGORY_NAMES[1:],
                            y=per_class_rate,
                            marker=dict(color=COLORS[1:],
                                        line=dict(color="#0a0e1a", width=1)),
                            text=[f"{v:.1f}%" for v in per_class_rate],
                            textposition="outside",
                            textfont=dict(color="#e2e8f0", size=10),
                        )],
                        layout=go.Layout(
                            title=dict(text="Detection Rate by Attack Class",
                                       font=dict(color="#94a3b8", size=13)),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#cbd5e1"),
                            yaxis=dict(range=[0, 115], showgrid=True,
                                       gridcolor="#1e293b",
                                       title="Detection Rate (%)",
                                       tickfont=dict(size=9)),
                            xaxis=dict(tickfont=dict(size=9)),
                            margin=dict(t=50, b=60, l=60, r=20),
                            bargap=0.35,
                        ),
                    ),
                    config={"displayModeBar": False},
                    style={"height": "300px"},
                )
            ]),

            html.Div(style=_card(flex=True), children=[
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Bar(
                            x=feat_vals[::-1],
                            y=feat_names[::-1],
                            orientation="h",
                            marker=dict(
                                color=feat_vals[::-1],
                                colorscale="Teal",
                                showscale=False,
                                line=dict(color="#0a0e1a", width=1),
                            ),
                        )],
                        layout=go.Layout(
                            title=dict(text="Feature Importance (Random Forest)",
                                       font=dict(color="#94a3b8", size=13)),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#cbd5e1"),
                            xaxis=dict(showgrid=True, gridcolor="#1e293b",
                                       tickfont=dict(size=9)),
                            yaxis=dict(tickfont=dict(size=9)),
                            margin=dict(t=50, b=20, l=165, r=20),
                        ),
                    ),
                    config={"displayModeBar": False},
                    style={"height": "300px"},
                )
            ]),
        ]),

        # ── Footer ──────────────────────────────────────────────────────────
        html.Div(
            "Isolation Forest (anomaly detection)  +  Random Forest (attack classification)  "
            "·  13 engineered features  ·  6-class output",
            style={
                "textAlign": "center", "fontSize": "11px",
                "color": "#334155", "letterSpacing": "1.5px",
                "borderTop": "1px solid #1e293b", "paddingTop": "14px",
            }
        ),
    ])

    return app


# ── small helpers ────────────────────────────────────────────────────────────
def _kpi(label, value, color):
    return html.Div(style={
        "flex": "1",
        "background": "#0f172a",
        "border": f"1px solid {color}30",
        "borderRadius": "6px",
        "padding": "18px 20px",
    }, children=[
        html.Div(label, style={
            "fontSize": "10px", "letterSpacing": "2px",
            "color": "#64748b", "marginBottom": "8px",
        }),
        html.Div(value, style={
            "fontSize": "30px", "fontWeight": "bold", "color": color,
        }),
    ])


def _card(flex=False):
    base = {
        "background": "#0f172a",
        "border": "1px solid #1e293b",
        "borderRadius": "6px",
        "padding": "10px",
    }
    if flex:
        base["flex"] = "1"
    return base


# ────────────────────────────────────────────────────────────────────────────
# MAIN — run everything, then launch dashboard
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("       CLOUD INTRUSION DETECTION SYSTEM — COMPLETE PIPELINE")
    print("="*70)

    # 1. Data
    train_df, test_df = load_data()

    # 2. Preprocessing
    X_train, y_train, X_test, y_test, _ = preprocess_pipeline(train_df, test_df)

    # 3. Isolation Forest
    iso = train_isolation_forest(X_train, y_train)
    evaluate_isolation_forest(iso, X_test, y_test)

    # 4. Random Forest
    rf = train_random_forest(X_train, y_train)
    evaluate_random_forest(rf, X_test, y_test)

    # 5. Dual-model evaluation
    results = evaluate_dual_model(iso, rf, X_test, y_test)

    # 6. Dashboard
    print("\n" + "="*70)
    print("STEP 6 — LAUNCHING DASHBOARD")
    print("="*70)
    print("  → http://127.0.0.1:8050")
    print("  Press Ctrl-C to stop.\n")

    app = build_dashboard(results, rf)
    app.run(debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
