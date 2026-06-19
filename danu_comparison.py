import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import stats
import os
import json
import pickle
import time

# ============================
# Google Drive setup
# ============================
DRIVE_SAVE_DIR = "/content/drive/MyDrive/lorenz96_experiment300"
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

def drive_path(filename):
    return os.path.join(DRIVE_SAVE_DIR, filename)

def save_seed_checkpoint(seed_idx, seed, model_hists, metrics, dataset_name):
    checkpoint = {
        "seed_idx":     seed_idx,
        "seed":         seed,
        "model_hists":  model_hists,
        "metrics":      metrics,
        "dataset_name": dataset_name,
        "timestamp":    time.time(),
    }
    fname = drive_path(f"seed_{seed_idx:02d}_seed{seed}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  ✓ Checkpoint saved → {fname}")
    return fname

def load_seed_checkpoint(seed_idx, seed):
    fname = drive_path(f"seed_{seed_idx:02d}_seed{seed}.pkl")
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            cp = pickle.load(f)
        print(f"  ↩ Resuming from checkpoint: {fname}")
        return cp
    return None

def save_final_results(results, dataset_name):
    tag = dataset_name.replace(" ", "_").replace("/", "-")
    fname = drive_path(f"final_results_{tag}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    summary = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
               for k, v in results.get("statistics", {}).items()}
    summary_fname = drive_path(f"final_summary_{tag}.json")
    with open(summary_fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Final results saved → {fname}")
    print(f"✓ Summary JSON saved  → {summary_fname}")

# ============================
# Global config
# ============================
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS  = 300
PATIENCE    = 50
BATCH_SIZE  = 32
HIDDEN_SIZE = 128
NUM_SEEDS   = 10
SEEDS       = list(range(NUM_SEEDS))

# Lorenz-96 specific
L96_N    = 8      # number of spatial dimensions
                  # K=4 compartments → 2 dims per compartment
L96_F    = 8    # forcing: F=8 is standard weakly-chaotic regime
L96_DT   = 0.05   # timestep — safe for RK4, would overflow Euler
SEQ_LEN  = 100    # shorter than Lorenz because L96 decorrelates faster
HORIZON  = 10     # 10 * 0.05 = 0.5 time units ≈ 83% of Lyapunov time
TRAJ_LEN = 50000  # longer trajectory for 8-dim system

# ============================
# Lorenz-96 generator — RK4
# ============================

def lorenz96_rhs(state, F=L96_F):
    """
    Vectorised RHS of L96.
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    np.roll handles cyclic boundary conditions without a Python loop.
    """
    return ((np.roll(state, -1) - np.roll(state, 2))
            * np.roll(state, 1) - state + F)


def lorenz96_step(state, F=L96_F, dt=L96_DT):
    """
    RK4 integration step.
    Replaces Euler — Euler with dt=0.05 overflows L96's quadratic
    nonlinearity within a few hundred steps.
    RK4 is stable for F<=16 at dt=0.05.
    """
    k1 = lorenz96_rhs(state,           F=F)
    k2 = lorenz96_rhs(state + dt/2*k1, F=F)
    k3 = lorenz96_rhs(state + dt/2*k2, F=F)
    k4 = lorenz96_rhs(state + dt*k3,   F=F)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def generate_lorenz96(T=TRAJ_LEN, N=L96_N, F=L96_F,
                      dt=L96_DT, warmup=2000):
    """
    Generate a normalised L96 trajectory of length T.
    Raises ValueError if the system diverges — catches bad
    hyperparameter combinations before they silently corrupt training.
    """
    state    = F * np.ones(N)
    state[0] += 0.01   # small asymmetric perturbation

    for _ in range(warmup):
        state = lorenz96_step(state, F=F, dt=dt)
        if not np.isfinite(state).all():
            raise ValueError(
                f"L96 diverged during warmup. "
                f"Reduce dt (current={dt}) or F (current={F}).")

    traj = []
    for _ in range(T):
        state = lorenz96_step(state, F=F, dt=dt)
        traj.append(state.copy())

    traj = np.array(traj)   # [T, N]
    if not np.isfinite(traj).all():
        raise ValueError("L96 trajectory contains NaN/Inf after warmup.")

    # Z-score per dimension
    traj = (traj - traj.mean(axis=0)) / (traj.std(axis=0) + 1e-8)
    return traj


# ============================
# Dataset
# ============================

class Lorenz96Dataset(Dataset):
    def __init__(self, traj, indices, seq_len=SEQ_LEN,
                 prediction_horizon=HORIZON):
        """
        Takes a pre-generated trajectory and explicit window indices.
        Train and val datasets share the trajectory but use
        non-overlapping index ranges — no leakage.
        """
        self.data    = []
        self.targets = []

        for i in indices:
            x = traj[i:i + seq_len]
            y = traj[i + seq_len + prediction_horizon - 1]
            self.data.append(x)
            self.targets.append(y)

        self.data    = torch.tensor(np.array(self.data),    dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def make_train_val_datasets(seed, seq_len=SEQ_LEN,
                             prediction_horizon=HORIZON,
                             N=L96_N, F=L96_F,
                             n_train=1200, n_val=300):
    np.random.seed(seed)
    traj      = generate_lorenz96(T=TRAJ_LEN, N=N, F=F)
    max_start = len(traj) - seq_len - prediction_horizon

    stride     = max(1, seq_len // 4)
    all_starts = list(range(0, max_start, stride))
    split      = int(0.8 * len(all_starts))

    train_pool = all_starts[:split]

    # ── FIX: enforce a clean gap before val starts ──────────────────────────
    # The last train window ends at:  train_pool[-1] + seq_len + horizon - 1
    # Val windows must start at least seq_len steps after that end point
    # so no input timestep in val was ever seen as an input in training.
    last_train_end = train_pool[-1] + seq_len + prediction_horizon
    first_val_start = last_train_end + seq_len          # full window gap

    val_pool = [s for s in all_starts[split:] if s >= first_val_start]

    if len(val_pool) < 50:
        # Not enough val windows after gap — extend trajectory or reduce seq_len
        raise ValueError(
            f"Only {len(val_pool)} val windows after gap. "
            f"Increase TRAJ_LEN or reduce SEQ_LEN.")

    np.random.shuffle(train_pool)
    np.random.shuffle(val_pool)
    train_idx = train_pool[:n_train]
    val_idx   = val_pool[:n_val]

    train_ds = Lorenz96Dataset(traj, train_idx,
                                seq_len=seq_len,
                                prediction_horizon=prediction_horizon)
    val_ds   = Lorenz96Dataset(traj, val_idx,
                                seq_len=seq_len,
                                prediction_horizon=prediction_horizon)
    return train_ds, val_ds



# ============================
# Models
# ============================

class TractableDendriticCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_comp    = num_compartments
        self.comp_size   = hidden_size // num_compartments
        assert hidden_size % num_compartments == 0
        self.W_in  = nn.ModuleList([nn.Linear(input_size, self.comp_size)
                                    for _ in range(num_compartments)])
        self.W_rec = nn.ModuleList([nn.Linear(hidden_size, self.comp_size)
                                    for _ in range(num_compartments)])
        self.integration = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        comp_outputs = []
        for i in range(self.num_comp):
            local_out = torch.tanh(self.W_in[i](x) + self.W_rec[i](h))
            comp_outputs.append(local_out)
        combined = torch.cat(comp_outputs, dim=1)
        return torch.tanh(self.integration(combined))


class TractableDendriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = TractableDendriticCell(input_size, hidden_size, num_compartments)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(h)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


class DANUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments=4):
        super().__init__()
        self.K = num_compartments
        self.H = hidden_size
        in_dim = input_size + hidden_size

        # Stack all compartments into single weight matrices
        self.W_E = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_I = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_g = nn.Linear(in_dim + hidden_size,
                             hidden_size * num_compartments)
        self.lambda_I = nn.Parameter(
            torch.ones(num_compartments, hidden_size) * 0.1)
        self.ln    = nn.LayerNorm(hidden_size)
        self.beta  = nn.Parameter(torch.ones(num_compartments, 1))
        self.theta = nn.Parameter(torch.zeros(num_compartments, 1))
        self.gamma = nn.Parameter(
            torch.ones(num_compartments, 1) * 0.1)
        self.W_q    = nn.Linear(hidden_size, hidden_size)
        self.W_out  = nn.Linear(hidden_size, hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        B  = x.size(0)
        xh = torch.cat([x, h], dim=1)   # [B, in_dim]

        # All compartments in one matmul
        E = self.W_E(xh).view(B, self.K, self.H)   # [B, K, H]
        I = self.W_I(xh).view(B, self.K, self.H)   # [B, K, H]
        u = E - self.lambda_I * I                   # [B, K, H]
        u = self.ln(u)                              # broadcast over K

        sig  = torch.sigmoid(self.beta  * (u - self.theta))
        nmda = self.gamma * (u**2) / (1.0 + u**2)
        d_tilde = sig + nmda                        # [B, K, H]

        # Gating
        d_bar   = d_tilde.mean(dim=1, keepdim=True)             # [B, 1, H]
        xh_dbar = torch.cat([xh, d_bar.squeeze(1)], dim=1)
        G       = torch.sigmoid(
            self.W_g(xh_dbar).view(B, self.K, self.H))          # [B, K, H]
        d_gated = G * d_tilde                                    # [B, K, H]

        # Attention
        q      = self.W_q(h).unsqueeze(2)                       # [B, H, 1]
        scores = torch.bmm(d_gated, q).squeeze(2) / (self.H**0.5)  # [B, K]
        alpha  = torch.softmax(scores, dim=1)                   # [B, K]
        attended = (alpha.unsqueeze(2) * d_gated).sum(dim=1)    # [B, H]

        h_new = torch.tanh(self.ln_out(self.W_out(attended) + h))
        return h_new


class DANU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DANUCell(input_size, hidden_size, num_compartments)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(h)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# ============================
# Training with early stopping — FIXED
# ============================

def train_model(model, train_loader, val_loader,
                max_epochs=MAX_EPOCHS, patience=PATIENCE,
                lr=1e-3, device="cpu",
                clip_grad=True, max_grad_norm=1.0):
    """
    Three fixes vs previous version:
    1. RK4 integration upstream means data is clean — but NaN guard
       now skips bad batches instead of breaking the epoch loop entirely
    2. optimizer.step() called before scheduler.step() (PyTorch requirement)
    3. Entire-epoch NaN detection with graceful recovery
    """
    model.to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs, eta_min=lr / 50)
    loss_fn   = nn.MSELoss()
    hist      = {"train_loss": [], "val_loss": []}

    best_val         = float('inf')
    patience_counter = 0
    best_state       = None
    stopped_epoch    = max_epochs

    for epoch in range(max_epochs):
        model.train()
        tl           = 0.0
        valid_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)

            if not torch.isfinite(loss):
                # Skip bad batch — don't update weights
                print(f"  WARNING: NaN/Inf loss at epoch {epoch} — skipping batch")
                continue

            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()   # Fix 2: optimizer before scheduler
            tl += loss.item()
            valid_batches += 1

        scheduler.step()   # Fix 2: scheduler after optimizer

        if valid_batches == 0:
            # Entire epoch was NaN — record and continue rather than crash
            print(f"  WARNING: entire epoch {epoch} had no valid batches")
            hist["train_loss"].append(float('nan'))
            hist["val_loss"].append(float('nan'))
            continue

        tl /= valid_batches

        model.eval()
        vl            = 0.0
        valid_val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                if torch.isfinite(loss):
                    vl += loss.item()
                    valid_val_batches += 1

        if valid_val_batches == 0:
            hist["train_loss"].append(tl)
            hist["val_loss"].append(float('nan'))
            continue

        vl /= valid_val_batches

        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)

        # Early stopping
        if vl < best_val:
            best_val         = vl
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopped_epoch = epoch + 1
                print(f"    Early stop at epoch {stopped_epoch} "
                      f"(best val={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist["stopped_epoch"] = stopped_epoch
    hist["best_val"]      = best_val
    return hist


# ============================
# Multi-seed experiment
# ============================

def run_comparison(dataset_name,
                   input_size, output_size, hidden_size=HIDDEN_SIZE,
                   max_epochs=MAX_EPOCHS, patience=PATIENCE,
                   batch_size=BATCH_SIZE, device=DEVICE, seeds=SEEDS):

    all_tractable_hist = []
    all_gru_hist       = []
    all_lstm_hist      = []
    all_danu_hist      = []
    final_metrics      = []

    print(f"\n{'='*80}")
    print(f"Running {dataset_name} — {len(seeds)} seeds")
    print(f"input_size={input_size}, output_size={output_size}, "
          f"hidden={hidden_size}")
    print(f"Early stopping: patience={patience}, max_epochs={max_epochs}")
    print(f"Checkpoints → {DRIVE_SAVE_DIR}")
    print(f"{'='*80}")

    for idx, seed in enumerate(seeds):
        print(f"\nSeed {idx+1}/{len(seeds)} (seed={seed})")

        cp = load_seed_checkpoint(idx, seed)
        if cp is not None:
            hists   = cp["model_hists"]
            metrics = cp["metrics"]
            all_tractable_hist.append(hists["tractable"])
            all_gru_hist.append(hists["gru"])
            all_lstm_hist.append(hists["lstm"])
            all_danu_hist.append(hists["danu"])
            final_metrics.append(metrics)
            print(f"  Skipped — loaded from checkpoint. "
                  f"DANU stopped @ epoch {hists['danu'].get('stopped_epoch','?')}, "
                  f"GRU @ {hists['gru'].get('stopped_epoch','?')}")
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate one trajectory per seed, split by position — no leakage
        train_dataset, val_dataset = make_train_val_datasets(
            seed=seed,
            seq_len=SEQ_LEN,
            prediction_horizon=HORIZON,
            N=L96_N,
            F=L96_F,
            n_train=1200,
            n_val=300,
        )
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        tractable = TractableDendriticRNN(input_size, hidden_size,
                                          output_size=output_size,
                                          num_compartments=4).to(device)
        gru       = GRUModel(input_size, hidden_size,
                             output_size=output_size).to(device)
        lstm      = LSTMModel(input_size, hidden_size,
                              output_size=output_size).to(device)
        danu      = DANU(input_size, hidden_size,
                         output_size=output_size,
                         num_compartments=4).to(device)

        print("  Training Tractable Dendritic RNN...")
        tractable_hist = train_model(
            tractable, train_loader, val_loader,
            max_epochs=max_epochs, patience=patience,
            lr=5e-4, device=device,
            clip_grad=True, max_grad_norm=1.0)
        print(f"    → stopped epoch {tractable_hist['stopped_epoch']}, "
              f"best val={tractable_hist['best_val']:.6f}")

        print("  Training GRU...")
        gru_hist = train_model(
            gru, train_loader, val_loader,
            max_epochs=max_epochs, patience=patience,
            lr=1e-4, device=device,
            clip_grad=True, max_grad_norm=1.0)
        print(f"    → stopped epoch {gru_hist['stopped_epoch']}, "
              f"best val={gru_hist['best_val']:.6f}")

        print("  Training LSTM...")
        lstm_hist = train_model(
            lstm, train_loader, val_loader,
            max_epochs=max_epochs, patience=patience,
            lr=5e-5, device=device,
            clip_grad=True, max_grad_norm=1.0)
        print(f"    → stopped epoch {lstm_hist['stopped_epoch']}, "
              f"best val={lstm_hist['best_val']:.6f}")

        print("  Training DANU...")
        danu_hist = train_model(
            danu, train_loader, val_loader,
            max_epochs=max_epochs, patience=patience,
            lr=5e-3, device=device,
            clip_grad=True, max_grad_norm=0.5)
        print(f"    → stopped epoch {danu_hist['stopped_epoch']}, "
              f"best val={danu_hist['best_val']:.6f}")
        '''
        danu_hist = {"train_loss": [], "val_loss": [],
             "stopped_epoch": 0, "best_val": float('nan')}
        '''
        t_best  = tractable_hist['best_val']
        g_best  = gru_hist['best_val']
        l_best  = lstm_hist['best_val']
        da_best = danu_hist['best_val']

        seed_metrics = {
            "tractable_best":           t_best,
            "gru_best":                 g_best,
            "lstm_best":                l_best,
            "danu_best":                da_best,
            "improvement_vs_lstm":      ((l_best  - da_best) / l_best)  * 100,
            "improvement_vs_tractable": ((t_best  - da_best) / t_best)  * 100,
            "improvement_vs_gru":       ((g_best  - da_best) / g_best)  * 100,
            "tractable_stopped_epoch":  tractable_hist['stopped_epoch'],
            "gru_stopped_epoch":        gru_hist['stopped_epoch'],
            "lstm_stopped_epoch":       lstm_hist['stopped_epoch'],
            "danu_stopped_epoch":       danu_hist['stopped_epoch'],
        }

        hists = {
            "tractable": tractable_hist,
            "gru":       gru_hist,
            "lstm":      lstm_hist,
            "danu":      danu_hist,
        }

        save_seed_checkpoint(idx, seed, hists, seed_metrics, dataset_name)

        all_tractable_hist.append(tractable_hist)
        all_gru_hist.append(gru_hist)
        all_lstm_hist.append(lstm_hist)
        all_danu_hist.append(danu_hist)
        final_metrics.append(seed_metrics)

        print(f"  DANU: {da_best:.6f} | Tractable: {t_best:.6f} | "
              f"GRU: {g_best:.6f} | LSTM: {l_best:.6f}")
        #print(f"  Tractable: {t_best:.6f} | GRU: {g_best:.6f} | LSTM: {l_best:.6f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def pad_and_stack(hist_list, key):
        arrays  = [np.array(h[key]) for h in hist_list]
        max_len = max(len(a) for a in arrays)
        padded  = [np.pad(a, (0, max_len - len(a)), mode='edge')
                   for a in arrays]
        return np.stack(padded, axis=0)

    results = {
        "tractable_train_mean": pad_and_stack(all_tractable_hist, 'train_loss').mean(axis=0),
        "tractable_train_std":  pad_and_stack(all_tractable_hist, 'train_loss').std(axis=0),
        "tractable_val_mean":   pad_and_stack(all_tractable_hist, 'val_loss').mean(axis=0),
        "tractable_val_std":    pad_and_stack(all_tractable_hist, 'val_loss').std(axis=0),
        "gru_train_mean":       pad_and_stack(all_gru_hist,       'train_loss').mean(axis=0),
        "gru_train_std":        pad_and_stack(all_gru_hist,       'train_loss').std(axis=0),
        "gru_val_mean":         pad_and_stack(all_gru_hist,       'val_loss').mean(axis=0),
        "gru_val_std":          pad_and_stack(all_gru_hist,       'val_loss').std(axis=0),
        "lstm_train_mean":      pad_and_stack(all_lstm_hist,      'train_loss').mean(axis=0),
        "lstm_train_std":       pad_and_stack(all_lstm_hist,      'train_loss').std(axis=0),
        "lstm_val_mean":        pad_and_stack(all_lstm_hist,      'val_loss').mean(axis=0),
        "lstm_val_std":         pad_and_stack(all_lstm_hist,      'val_loss').std(axis=0),
        "danu_train_mean":      pad_and_stack(all_danu_hist,      'train_loss').mean(axis=0),
        "danu_train_std":       pad_and_stack(all_danu_hist,      'train_loss').std(axis=0),
        "danu_val_mean":        pad_and_stack(all_danu_hist,      'val_loss').mean(axis=0),
        "danu_val_std":         pad_and_stack(all_danu_hist,      'val_loss').std(axis=0),
        "final_metrics":        final_metrics,
    }

    tractable_bests = [m['tractable_best'] for m in final_metrics]
    gru_bests       = [m['gru_best']       for m in final_metrics]
    lstm_bests      = [m['lstm_best']      for m in final_metrics]
    danu_bests      = [m['danu_best']      for m in final_metrics]

    improvements_vs_lstm      = [m['improvement_vs_lstm']      for m in final_metrics]
    improvements_vs_tractable = [m['improvement_vs_tractable'] for m in final_metrics]
    improvements_vs_gru       = [m['improvement_vs_gru']       for m in final_metrics]

    _, p_danu_vs_lstm      = stats.ttest_rel(danu_bests, lstm_bests)
    _, p_danu_vs_gru       = stats.ttest_rel(danu_bests, gru_bests)
    _, p_danu_vs_tractable = stats.ttest_rel(danu_bests, tractable_bests)
    _, p_tractable_vs_lstm = stats.ttest_rel(tractable_bests, lstm_bests)
    _, p_gru_vs_lstm       = stats.ttest_rel(gru_bests, lstm_bests)

    def sig(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS — {dataset_name}")
    print(f"{'='*80}")
    print(f"\nBest Validation Loss (mean ± std):")
    print(f"  DANU (proposed):               {np.mean(danu_bests):.6f} ± {np.std(danu_bests):.6f}")
    print(f"  Tractable (Passive Dendrites): {np.mean(tractable_bests):.6f} ± {np.std(tractable_bests):.6f}")
    print(f"  GRU (Baseline):                {np.mean(gru_bests):.6f} ± {np.std(gru_bests):.6f}")
    print(f"  LSTM (Baseline):               {np.mean(lstm_bests):.6f} ± {np.std(lstm_bests):.6f}")

    print(f"\nActual epochs to convergence (mean ± std):")
    for name, key in [('DANU',      'danu_stopped_epoch'),
                      ('GRU',       'gru_stopped_epoch'),
                      ('LSTM',      'lstm_stopped_epoch'),
                      ('Tractable', 'tractable_stopped_epoch')]:
        vals = [m[key] for m in final_metrics]
        print(f"  {name:10s}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")

    print(f"\nRelative Improvement (DANU vs):")
    print(f"  DANU vs LSTM:      {np.mean(improvements_vs_lstm):+.2f}% ± {np.std(improvements_vs_lstm):.2f}%")
    print(f"  DANU vs Tractable: {np.mean(improvements_vs_tractable):+.2f}% ± {np.std(improvements_vs_tractable):.2f}%")
    print(f"  DANU vs GRU:       {np.mean(improvements_vs_gru):+.2f}% ± {np.std(improvements_vs_gru):.2f}%")

    print(f"\nStatistical Significance (paired t-test):")
    print(f"  DANU vs LSTM:      p = {p_danu_vs_lstm:.4f} {sig(p_danu_vs_lstm)}")
    print(f"  DANU vs GRU:       p = {p_danu_vs_gru:.4f} {sig(p_danu_vs_gru)}")
    print(f"  DANU vs Tractable: p = {p_danu_vs_tractable:.4f} {sig(p_danu_vs_tractable)}")
    print(f"  Tractable vs LSTM: p = {p_tractable_vs_lstm:.4f} {sig(p_tractable_vs_lstm)}")
    print(f"  GRU vs LSTM:       p = {p_gru_vs_lstm:.4f} {sig(p_gru_vs_lstm)}")

    results['statistics'] = {
        'p_danu_vs_lstm':                p_danu_vs_lstm,
        'p_danu_vs_gru':                 p_danu_vs_gru,
        'p_danu_vs_tractable':           p_danu_vs_tractable,
        'p_tractable_vs_lstm':           p_tractable_vs_lstm,
        'p_gru_vs_lstm':                 p_gru_vs_lstm,
        'improvement_vs_lstm_mean':      np.mean(improvements_vs_lstm),
        'improvement_vs_lstm_std':       np.std(improvements_vs_lstm),
        'improvement_vs_tractable_mean': np.mean(improvements_vs_tractable),
        'improvement_vs_tractable_std':  np.std(improvements_vs_tractable),
        'improvement_vs_gru_mean':       np.mean(improvements_vs_gru),
        'improvement_vs_gru_std':        np.std(improvements_vs_gru),
    }

    save_final_results(results, dataset_name)
    return results


# ============================
# Plotting
# ============================

def visualize_results(results, title, save_path=None):
    epochs = np.arange(1, len(results["danu_train_mean"]) + 1)

    COLOR_DANU      = '#1B4332'
    COLOR_TRACTABLE = '#F18F01'
    COLOR_GRU       = '#06A77D'
    COLOR_LSTM      = '#A23B72'

    models = [
        ("danu",      "DANU",      COLOR_DANU),
        ("tractable", "Tractable", COLOR_TRACTABLE),
        ("gru",       "GRU",       COLOR_GRU),
        ("lstm",      "LSTM",      COLOR_LSTM),
    ]

    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig1.suptitle(title, fontsize=15, fontweight='bold', y=0.999)

    panel_cfg = [
        (axes[0, 0], "train", "log",    "Training Loss  — log scale (mean ± std)"),
        (axes[0, 1], "val",   "log",    "Validation Loss — log scale (mean ± std)"),
        (axes[1, 0], "train", "linear", "Training Loss  — linear scale (mean ± std)"),
        (axes[1, 1], "val",   "linear", "Validation Loss — linear scale (mean ± std)"),
    ]

    for ax, split, scale, panel_title in panel_cfg:
        for key, label, color in models:
            mean = results[f"{key}_{split}_mean"]
            std  = results[f"{key}_{split}_std"]
            ax.plot(epochs, mean, label=label, color=color, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.20, color=color)
        ax.set_title(panel_title, fontsize=12, fontweight='bold')
        ax.set_yscale(scale)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("MSE Loss" + (" (log)" if scale == "log" else ""),
                      fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    final_metrics  = results['final_metrics']
    danu_vals      = [m['danu_best']      for m in final_metrics]
    tractable_vals = [m['tractable_best'] for m in final_metrics]
    gru_vals       = [m['gru_best']       for m in final_metrics]
    lstm_vals      = [m['lstm_best']      for m in final_metrics]

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(title + " — Final Performance", fontsize=14, fontweight='bold')

    x_pos  = np.arange(4)
    means  = [np.mean(danu_vals), np.mean(tractable_vals),
               np.mean(gru_vals),  np.mean(lstm_vals)]
    stds   = [np.std(danu_vals),  np.std(tractable_vals),
               np.std(gru_vals),   np.std(lstm_vals)]
    colors = [COLOR_DANU, COLOR_TRACTABLE, COLOR_GRU, COLOR_LSTM]

    axes2[0].bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.75)
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels(['DANU', 'Tractable', 'GRU', 'LSTM'], fontsize=11)
    axes2[0].set_ylabel('Best Validation Loss', fontsize=11)
    axes2[0].set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    axes2[0].grid(alpha=0.3, axis='y')

    if 'statistics' in results:
        y_max = max(m + s for m, s in zip(means, stds)) * 1.05
        for (i, j), level, key in [
            ((0, 3), 1.00, 'p_danu_vs_lstm'),
            ((0, 2), 0.88, 'p_danu_vs_gru'),
            ((0, 1), 0.76, 'p_danu_vs_tractable'),
        ]:
            p_val = results['statistics'][key]
            stars = ('***' if p_val < 0.001 else '**' if p_val < 0.01
                     else '*' if p_val < 0.05 else 'n.s.')
            y = y_max * level
            axes2[0].plot([i, j], [y, y], 'k-', linewidth=1)
            axes2[0].text((i + j) / 2, y * 1.015, stars,
                          ha='center', fontsize=11, fontweight='bold')

    improvements_lstm      = [m['improvement_vs_lstm']      for m in final_metrics]
    improvements_tractable = [m['improvement_vs_tractable'] for m in final_metrics]
    improvements_gru       = [m['improvement_vs_gru']       for m in final_metrics]

    x_imp     = np.arange(3)
    imp_means = [np.mean(improvements_lstm),
                 np.mean(improvements_tractable),
                 np.mean(improvements_gru)]
    imp_stds  = [np.std(improvements_lstm),
                 np.std(improvements_tractable),
                 np.std(improvements_gru)]

    axes2[1].bar(x_imp, imp_means, yerr=imp_stds, capsize=5,
                 color=[COLOR_LSTM, COLOR_TRACTABLE, COLOR_GRU], alpha=0.75)
    axes2[1].set_xticks(x_imp)
    axes2[1].set_xticklabels(['DANU vs\nLSTM', 'DANU vs\nTractable',
                               'DANU vs\nGRU'], fontsize=11)
    axes2[1].set_ylabel('Improvement (%)', fontsize=11)
    axes2[1].set_title('DANU Advantage (mean ± std)', fontsize=12, fontweight='bold')
    axes2[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes2[1].grid(alpha=0.3, axis='y')

    fig2.tight_layout()

    if save_path:
        base         = save_path.replace('.png', '')
        curves_path  = base + '_curves.png'
        bars_path    = base + '_bars.png'
        drive_curves = drive_path(os.path.basename(curves_path))
        drive_bars   = drive_path(os.path.basename(bars_path))
        for fig, lpath, dpath in [
            (fig1, curves_path, drive_curves),
            (fig2, bars_path,   drive_bars),
        ]:
            fig.savefig(lpath, dpi=300, bbox_inches='tight')
            fig.savefig(dpath, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved locally:  {curves_path}  |  {bars_path}")
        print(f"Plots saved to Drive: {drive_curves}  |  {drive_bars}")

    plt.show()


# ============================
# Reload helper
# ============================

def reload_all_checkpoints(dataset_name, seeds=SEEDS):
    all_hists     = {"tractable": [], "gru": [], "lstm": [], "danu": []}
    final_metrics = []

    for idx, seed in enumerate(seeds):
        cp = load_seed_checkpoint(idx, seed)
        if cp is None:
            raise FileNotFoundError(
                f"No checkpoint for seed_idx={idx}, seed={seed}")
        for model_name in all_hists:
            all_hists[model_name].append(cp["model_hists"][model_name])
        final_metrics.append(cp["metrics"])

    def pad_and_stack(hist_list, key):
        arrays  = [np.array(h[key]) for h in hist_list]
        max_len = max(len(a) for a in arrays)
        padded  = [np.pad(a, (0, max_len - len(a)), mode='edge')
                   for a in arrays]
        return np.stack(padded, axis=0)

    results = {}
    for model_name, hist_list in all_hists.items():
        results[f"{model_name}_train_mean"] = pad_and_stack(hist_list, 'train_loss').mean(axis=0)
        results[f"{model_name}_train_std"]  = pad_and_stack(hist_list, 'train_loss').std(axis=0)
        results[f"{model_name}_val_mean"]   = pad_and_stack(hist_list, 'val_loss').mean(axis=0)
        results[f"{model_name}_val_std"]    = pad_and_stack(hist_list, 'val_loss').std(axis=0)
    results["final_metrics"] = final_metrics

    print(f"Reloaded {len(seeds)} checkpoints from {DRIVE_SAVE_DIR}")
    return results


# ============================
# Main
# ============================

if __name__ == "__main__":

    # from google.colab import drive
    # drive.mount('/content/drive')

    print("="*80)
    print("DANU vs Tractable vs GRU vs LSTM  —  Lorenz-96")
    print(f"L96:  N={L96_N}, F={L96_F}, dt={L96_DT}  (RK4 integration)")
    print(f"Task: seq_len={SEQ_LEN}, horizon={HORIZON}, hidden={HIDDEN_SIZE}")
    print(f"Train: max_epochs={MAX_EPOCHS}, patience={PATIENCE}, lr=1e-3 (all)")
    print(f"Saving to: {DRIVE_SAVE_DIR}")
    print("="*80)

    DATASET_NAME = (f"Lorenz-96 N={L96_N} F={L96_F} "
                    f"— seq={SEQ_LEN}, h={HORIZON}")

    results = run_comparison(
        DATASET_NAME,
        input_size=L96_N,
        output_size=L96_N,
        hidden_size=HIDDEN_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        seeds=SEEDS,
    )

    visualize_results(
        results,
        f"Lorenz-96 (N={L96_N}, F={L96_F}): DANU vs Tractable vs GRU vs LSTM "
        f"— seq={SEQ_LEN}, h={HORIZON} (10 seeds, early stop, RK4)",
        drive_path("lorenz96_results.png"))

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print(f"All results saved to: {DRIVE_SAVE_DIR}")
    print("="*80)
