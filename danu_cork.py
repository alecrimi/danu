"""
DANU vs TractableDendriticRNN vs GRU vs LSTM
Cork Neonatal EEG Dataset — EDF format
Stratified by Grade (1, 2, 3, 4) at 200 Hz sampling frequency

Hypothesis: DANU outperforms baselines especially at Grade 3 & 4
            (more severe pathology → richer nonlinear dynamics).

Data layout:
    EDF_format/
        ID01_epoch1.edf
        ID02_epoch1.edf
        ...
        metadata.csv   ← columns: subject_id, grade, sampling_freq, ...
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR        = "EDF_format"
METADATA_FILE   = os.path.join(DATA_DIR, "metadata.csv")
TARGET_FS       = 200          # Hz — only subjects recorded at this fs
GRADES          = [1, 2, 3, 4] # grades to include
SEQ_LEN         = 256          # input window (1.28 s at 200 Hz)
HORIZON         = 50           # samples ahead to predict
HIDDEN_SIZE     = 128
NUM_SEEDS       = 5            # seeds per grade comparison
MAX_EPOCHS      = 300
PATIENCE        = 30
BATCH_SIZE      = 32
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR        = "cork_danu_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# EDF LOADING
# ─────────────────────────────────────────────

def load_edf(path):
    """
    Load an EDF file and return (data, sfreq).
    data shape: [n_channels, n_samples]
    """
    if MNE_AVAILABLE:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        return raw.get_data(), raw.info["sfreq"]
    else:
        raise ImportError("mne is required: pip install mne")


def resample_signal(data, orig_fs, target_fs):
    """
    Simple resample using scipy — keeps all channels.
    data: [n_channels, n_samples]
    """
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(target_fs), int(orig_fs))
    up, down = int(target_fs) // g, int(orig_fs) // g
    resampled = resample_poly(data, up, down, axis=1)
    return resampled


def load_subjects(data_dir, metadata_file, target_fs=200, grades=None):
    """
    Returns a dict: { grade -> list of np.ndarray [n_channels, n_samples] }
    Only subjects whose sampling_freq == target_fs are included.
    """
    meta = pd.read_csv(metadata_file)
    # normalise column names to lowercase
    meta.columns = [c.strip().lower() for c in meta.columns]

    # filter by sampling frequency
    meta_fs = meta[meta["sampling_freq"] == target_fs].copy()
    print(f"[Data] {len(meta_fs)} subjects at {target_fs} Hz "
          f"(out of {len(meta)} total)")

    if grades is not None:
        meta_fs = meta_fs[meta_fs["grade"].isin(grades)]
        print(f"[Data] {len(meta_fs)} subjects in grades {grades}")

    grade_data = {g: [] for g in (grades or meta_fs["grade"].unique())}

    # try common subject_id column names
    id_col = None
    for c in ["subject_id", "id", "subject", "subjectid", "participant_id"]:
        if c in meta_fs.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError(f"Cannot find subject ID column. "
                         f"Columns found: {list(meta_fs.columns)}")

    loaded, skipped = 0, 0
    for _, row in meta_fs.iterrows():
        subj_id = str(row[id_col]).strip()
        grade   = int(row["grade"])
        fs      = float(row["sampling_freq"])

        # try several filename patterns
        candidates = [
            os.path.join(data_dir, f"{subj_id}_epoch1.edf"),
            os.path.join(data_dir, f"{subj_id}.edf"),
            os.path.join(data_dir, f"{subj_id}_epoch1.EDF"),
        ]
        edf_path = next((p for p in candidates if os.path.exists(p)), None)
        if edf_path is None:
            print(f"  [WARN] EDF not found for {subj_id} — skipping")
            skipped += 1
            continue

        try:
            data, sfreq = load_edf(edf_path)
            # resample if needed (shouldn't be, but just in case)
            if abs(sfreq - target_fs) > 0.5:
                data = resample_signal(data, sfreq, target_fs)
            # z-score each channel independently
            data = (data - data.mean(axis=1, keepdims=True)) / \
                   (data.std(axis=1, keepdims=True) + 1e-8)
            grade_data[grade].append(data)
            loaded += 1
        except Exception as e:
            print(f"  [WARN] Error loading {edf_path}: {e}")
            skipped += 1

    print(f"[Data] Loaded {loaded} subjects, skipped {skipped}")
    for g, slist in grade_data.items():
        print(f"  Grade {g}: {len(slist)} subjects")
    return grade_data


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class EEGWindowDataset(Dataset):
    """
    Sliding-window dataset over a list of EEG recordings.
    Input:  [SEQ_LEN, n_channels]
    Target: [n_channels]  (signal at t + HORIZON)
    """
    def __init__(self, recordings, seq_len=SEQ_LEN, horizon=HORIZON,
                 stride=None):
        self.samples = []
        self.targets = []
        stride = stride or max(1, seq_len // 4)
        for rec in recordings:   # rec: [n_ch, n_samples]
            n_ch, T = rec.shape
            for start in range(0, T - seq_len - horizon, stride):
                x = rec[:, start : start + seq_len].T          # [SEQ_LEN, n_ch]
                y = rec[:, start + seq_len + horizon - 1]      # [n_ch]
                self.samples.append(x.astype(np.float32))
                self.targets.append(y.astype(np.float32))

        self.samples = torch.from_numpy(np.stack(self.samples))
        self.targets = torch.from_numpy(np.stack(self.targets))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


def make_loaders(recordings, seed, seq_len=SEQ_LEN, horizon=HORIZON,
                 batch_size=BATCH_SIZE, val_ratio=0.2):
    """
    Split recordings list into train/val by subject (not by window)
    to prevent data leakage.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recordings))
    n_val = max(1, int(len(recordings) * val_ratio))
    val_recs   = [recordings[i] for i in idx[:n_val]]
    train_recs = [recordings[i] for i in idx[n_val:]]

    train_ds = EEGWindowDataset(train_recs, seq_len=seq_len, horizon=horizon)
    val_ds   = EEGWindowDataset(val_recs,   seq_len=seq_len, horizon=horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE == "cuda"))
    return train_loader, val_loader


# ─────────────────────────────────────────────
# MODELS  (unchanged from original DANU paper)
# ─────────────────────────────────────────────

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
        parts = [torch.tanh(self.W_in[i](x) + self.W_rec[i](h))
                 for i in range(self.num_comp)]
        return torch.tanh(self.integration(torch.cat(parts, dim=1)))


class TractableDendriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = TractableDendriticCell(input_size, hidden_size,
                                           num_compartments)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(h)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class DANUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments=4):
        super().__init__()
        self.K = num_compartments
        self.H = hidden_size
        in_dim = input_size + hidden_size
        self.W_E   = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_I   = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_g   = nn.Linear(in_dim + hidden_size,
                               hidden_size * num_compartments)
        self.lambda_I = nn.Parameter(
            torch.ones(num_compartments, hidden_size) * 0.1)
        self.ln    = nn.LayerNorm(hidden_size)
        self.beta  = nn.Parameter(torch.ones(num_compartments, 1))
        self.theta = nn.Parameter(torch.zeros(num_compartments, 1))
        self.gamma = nn.Parameter(torch.ones(num_compartments, 1) * 0.1)
        self.W_q   = nn.Linear(hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        B    = x.size(0)
        xh   = torch.cat([x, h], dim=1)
        E    = self.W_E(xh).view(B, self.K, self.H)
        I    = self.W_I(xh).view(B, self.K, self.H)
        u    = self.ln(E - self.lambda_I * I)
        sig  = torch.sigmoid(self.beta * (u - self.theta))
        nmda = self.gamma * (u ** 2) / (1.0 + u ** 2)
        d_tilde = sig + nmda
        d_bar   = d_tilde.mean(dim=1, keepdim=True)
        xh_dbar = torch.cat([xh, d_bar.squeeze(1)], dim=1)
        G       = torch.sigmoid(
            self.W_g(xh_dbar).view(B, self.K, self.H))
        d_gated = G * d_tilde
        q       = self.W_q(h).unsqueeze(2)
        scores  = torch.bmm(d_gated, q).squeeze(2) / (self.H ** 0.5)
        alpha   = torch.softmax(scores, dim=1)
        attended = (alpha.unsqueeze(2) * d_gated).sum(dim=1)
        return torch.tanh(self.ln_out(self.W_out(attended) + h))


class DANU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_compartments=4):
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


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_model(model, train_loader, val_loader,
                max_epochs=MAX_EPOCHS, patience=PATIENCE,
                lr=1e-3, clip=1.0):
    model.to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs, eta_min=lr / 50)
    loss_fn = nn.MSELoss()

    best_val, patience_ctr, best_state = float("inf"), 0, None
    hist = {"train": [], "val": [], "stopped": max_epochs, "best_val": float("inf")}

    for epoch in range(max_epochs):
        model.train()
        tl, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tl += loss.item(); nb += 1
        opt.step(); sched.step()

        model.eval()
        vl, nvb = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = loss_fn(model(x), y)
                if torch.isfinite(loss):
                    vl += loss.item(); nvb += 1

        tl = tl / nb   if nb   else float("nan")
        vl = vl / nvb  if nvb  else float("nan")
        hist["train"].append(tl)
        hist["val"].append(vl)

        if np.isfinite(vl) and vl < best_val:
            best_val = vl; patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                hist["stopped"] = epoch + 1
                break

    if best_state:
        model.load_state_dict(best_state)
    hist["best_val"] = best_val
    return hist


# ─────────────────────────────────────────────
# PER-GRADE EXPERIMENT
# ─────────────────────────────────────────────

def run_grade_experiment(grade, recordings, n_seeds=NUM_SEEDS):
    """
    Run multi-seed comparison for a single grade group.
    Returns dict with per-seed metrics.
    """
    n_ch = recordings[0].shape[0]   # number of EEG channels
    print(f"\n{'='*70}")
    print(f"  Grade {grade} | {len(recordings)} subjects | "
          f"{n_ch} channels | {n_seeds} seeds")
    print(f"{'='*70}")

    if len(recordings) < 3:
        print(f"  [SKIP] Not enough subjects for Grade {grade} "
              f"(need ≥ 3, got {len(recordings)})")
        return None

    all_metrics = []

    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        train_loader, val_loader = make_loaders(recordings, seed)

        print(f"\n  Seed {seed+1}/{n_seeds}  "
              f"| train windows={len(train_loader.dataset)} "
              f"| val windows={len(val_loader.dataset)}")

        models_cfg = [
            ("DANU",      DANU(n_ch, HIDDEN_SIZE, n_ch, 4),       5e-3, 0.5),
            ("Tractable", TractableDendriticRNN(n_ch, HIDDEN_SIZE, n_ch, 4), 5e-4, 1.0),
            ("GRU",       GRUModel(n_ch, HIDDEN_SIZE, n_ch),       1e-4, 1.0),
            ("LSTM",      LSTMModel(n_ch, HIDDEN_SIZE, n_ch),      5e-5, 1.0),
        ]

        seed_result = {}
        for name, model, lr, clip in models_cfg:
            print(f"    Training {name}...", end="", flush=True)
            h = train_model(model, train_loader, val_loader,
                            lr=lr, clip=clip)
            seed_result[name] = h["best_val"]
            print(f"  best_val={h['best_val']:.6f}  "
                  f"(stopped ep {h['stopped']})")

        all_metrics.append(seed_result)

    return all_metrics


# ─────────────────────────────────────────────
# STATISTICAL SUMMARY
# ─────────────────────────────────────────────

def summarise(all_metrics):
    """
    Compute mean ± std and paired t-test (DANU vs each baseline).
    """
    names = ["DANU", "Tractable", "GRU", "LSTM"]
    vals  = {n: [m[n] for m in all_metrics] for n in names}

    summary = {}
    for n in names:
        summary[n] = {"mean": np.mean(vals[n]), "std": np.std(vals[n])}

    for baseline in ["Tractable", "GRU", "LSTM"]:
        _, p = stats.ttest_rel(vals["DANU"], vals[baseline])
        imp  = np.mean(
            [(b - d) / b * 100 for d, b in
             zip(vals["DANU"], vals[baseline])])
        summary[f"p_danu_vs_{baseline}"] = p
        summary[f"imp_vs_{baseline}"]    = imp

    return summary, vals


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

COLORS = {
    "DANU":      "#1B4332",
    "Tractable": "#F18F01",
    "GRU":       "#06A77D",
    "LSTM":      "#A23B72",
}

def plot_grade_comparison(grade_results, save_path=None):
    """
    Two-panel figure:
      Left:  Bar chart — mean best-val loss per model, per grade
      Right: DANU improvement (%) over each baseline, per grade
    """
    grades = sorted(grade_results.keys())
    models = ["DANU", "Tractable", "GRU", "LSTM"]
    baselines = ["Tractable", "GRU", "LSTM"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "DANU vs Baselines — Cork Neonatal EEG\n"
        "Stratified by Pathology Grade (200 Hz)",
        fontsize=14, fontweight="bold")

    # ── Left: grouped bar chart ──────────────────────────────────────────
    ax = axes[0]
    x      = np.arange(len(grades))
    width  = 0.18
    offset = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * width

    for i, model in enumerate(models):
        means = [grade_results[g]["summary"][model]["mean"] for g in grades]
        stds  = [grade_results[g]["summary"][model]["std"]  for g in grades]
        ax.bar(x + offset[i], means, width,
               yerr=stds, capsize=4,
               color=COLORS[model], alpha=0.82, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Grade {g}" for g in grades], fontsize=11)
    ax.set_ylabel("Best Validation MSE Loss", fontsize=11)
    ax.set_title("Prediction Error by Grade", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # Add significance stars above DANU bars
    for gi, g in enumerate(grades):
        s = grade_results[g]["summary"]
        # choose the lowest p among baselines for the annotation
        p_vals = [s[f"p_danu_vs_{b}"] for b in baselines]
        best_p = min(p_vals)
        danu_mean = s["DANU"]["mean"]
        danu_std  = s["DANU"]["std"]
        ax.text(x[gi] + offset[0], danu_mean + danu_std + 0.002,
                sig_stars(best_p), ha="center", fontsize=12,
                color=COLORS["DANU"], fontweight="bold")

    # ── Right: improvement (%) ────────────────────────────────────────────
    ax2 = axes[1]
    baseline_colors = [COLORS[b] for b in baselines]

    for bi, baseline in enumerate(baselines):
        imps = [grade_results[g]["summary"][f"imp_vs_{baseline}"]
                for g in grades]
        ax2.plot(grades, imps, marker="o", linewidth=2,
                 color=COLORS[baseline], label=f"vs {baseline}")
        ax2.fill_between(grades,
                         [i - 2 for i in imps],
                         [i + 2 for i in imps],
                         alpha=0.1, color=COLORS[baseline])

    ax2.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axvspan(2.5, 4.5, alpha=0.07, color="#1B4332",
                label="Grade 3–4 (hypothesis zone)")
    ax2.set_xticks(grades)
    ax2.set_xticklabels([f"Grade {g}" for g in grades], fontsize=11)
    ax2.set_ylabel("DANU Improvement (%)\n(positive = DANU wins)", fontsize=11)
    ax2.set_title("DANU Advantage vs Pathology Severity", fontsize=12,
                  fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n[Plot] Saved → {save_path}")
    plt.close()


def print_summary_table(grade_results):
    grades  = sorted(grade_results.keys())
    models  = ["DANU", "Tractable", "GRU", "LSTM"]
    baselines = ["Tractable", "GRU", "LSTM"]

    print("\n" + "="*80)
    print("  RESULTS SUMMARY — Cork Neonatal EEG — DANU Comparison")
    print("="*80)

    for g in grades:
        s = grade_results[g]["summary"]
        print(f"\n  ── Grade {g} "
              f"({'hypothesis zone' if g >= 3 else 'control zone'}) ──")
        for model in models:
            print(f"    {model:12s}  "
                  f"MSE = {s[model]['mean']:.6f} ± {s[model]['std']:.6f}")
        print()
        for b in baselines:
            p   = s[f"p_danu_vs_{b}"]
            imp = s[f"imp_vs_{b}"]
            print(f"    DANU vs {b:10s}  "
                  f"improvement = {imp:+.2f}%   "
                  f"p = {p:.4f} {sig_stars(p)}")

    print("\n" + "="*80)
    print("  HYPOTHESIS CHECK: Does DANU gain more at Grade 3 & 4?")
    print("="*80)
    for b in baselines:
        low  = np.mean([grade_results[g]["summary"][f"imp_vs_{b}"]
                        for g in grades if g <= 2])
        high = np.mean([grade_results[g]["summary"][f"imp_vs_{b}"]
                        for g in grades if g >= 3])
        print(f"  vs {b:10s}  "
              f"Grade 1–2: {low:+.2f}%   Grade 3–4: {high:+.2f}%   "
              f"{'✓ CONFIRMED' if high > low else '✗ NOT confirmed'}")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*70)
    print("  DANU Cork EEG Experiment")
    print(f"  Target fs: {TARGET_FS} Hz | Grades: {GRADES}")
    print(f"  SEQ_LEN={SEQ_LEN}  HORIZON={HORIZON}  "
          f"HIDDEN={HIDDEN_SIZE}  SEEDS={NUM_SEEDS}")
    print(f"  Device: {DEVICE}")
    print("="*70)

    # ── Load data ──────────────────────────────────────────────────────────
    grade_data = load_subjects(DATA_DIR, METADATA_FILE,
                               target_fs=TARGET_FS, grades=GRADES)

    grade_results = {}

    for grade in GRADES:
        recordings = grade_data.get(grade, [])
        metrics = run_grade_experiment(grade, recordings, n_seeds=NUM_SEEDS)

        if metrics is None:
            print(f"  [SKIP] Grade {grade} skipped (insufficient data)")
            continue

        summary, vals = summarise(metrics)
        grade_results[grade] = {"metrics": metrics,
                                 "summary": summary,
                                 "vals":    vals}

    if not grade_results:
        print("\n[ERROR] No grades had sufficient data. "
              "Check metadata.csv and EDF files.")
        return

    # ── Print results ──────────────────────────────────────────────────────
    print_summary_table(grade_results)

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_path = os.path.join(SAVE_DIR, "cork_danu_comparison.png")
    plot_grade_comparison(grade_results, save_path=plot_path)

    print(f"\n[Done] Results saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
