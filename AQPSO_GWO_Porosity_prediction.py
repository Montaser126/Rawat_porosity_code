"""
AQPSO-GWO Hybrid Optimization for PHIE Prediction - GPU ACCELERATED
"""
# ==============================================================================
# CELL 1: Imports and Configuration
# ==============================================================================
import os
import sys
import warnings
import json
import pickle
import glob
import time
import gc
import atexit
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, AdaBoostRegressor)

# OPTIMIZATION: joblib for parallel objective evaluation
from joblib import Parallel, delayed

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# GPU / DEVICE SETUP
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'='*70}")
print(f"DEVICE: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
print(f"{'='*70}")

# ==============================================================================
# GPU-ACCELERATED ML: cuML for GPU-based sklearn equivalents
# ==============================================================================
try:
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.ensemble import ExtraTreesRegressor as cuET
    CUML_AVAILABLE = True
    print("cuML GPU: ENABLED - RF and ET will use GPU")
except ImportError:
    CUML_AVAILABLE = False
    print("cuML: Not available - using CPU sklearn for RF/ET")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

print("All imports completed successfully")

# ==============================================================================
# CONFIGURATION - OPTIMIZED PARAMETERS
# ==============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

INPUT_PATH   = r'C:\Users\ASUS\Desktop\1.4\Combined_Cleaned'
RESULTS_PATH = r'C:\Users\ASUS\Desktop\1.4\Results_AQPSO_GWO_RandomSplit'

TARGET_COLUMN = 'PHIE'
WELL_COL      = 'WELLNAME'
DEPTH_COL     = 'DEPTH'

# OPTIMIZATION
default_cv_folds = 3  
opt_n_agents = 5     
opt_max_iter = 10      
opt_patience = 3      
dl_opt_epochs = 20    
dl_base_epochs = 20  
dl_batch_size = 256    

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir    = os.path.join(RESULTS_PATH, f'run_{timestamp}_FAST')
plots_dir  = os.path.join(run_dir, 'plots')
models_dir = os.path.join(run_dir, 'saved_models')
for d in [run_dir, plots_dir, models_dir]:
    os.makedirs(d, exist_ok=True)
print(f"Results will be saved to: {run_dir}")

# ==============================================================================
# CHECKPOINTING - Crash Recovery
# ==============================================================================
baseline_results = {}
optimized_results = {}

def save_checkpoint():
    """Save current progress to resume if interrupted."""
    try:
        checkpoint = {
            'baseline_results': {
                k: {kk: vv for kk, vv in v.items() if kk != 'model'}
                for k, v in baseline_results.items()
            },
            'optimized_results_keys': list(optimized_results.keys()),
            'timestamp': datetime.now().isoformat()
        }
        cp_path = os.path.join(run_dir, 'checkpoint.json')
        with open(cp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")

atexit.register(save_checkpoint)
print("Checkpointing enabled (auto-saves on exit)")


# ==============================================================================
# CELL 2: OPTIMIZERS (OPTIMIZED WITH PARALLEL EVALUATION)
# ==============================================================================
class CachedObjective:
    """Cache objective function evaluations to avoid redundant computation."""
    def __init__(self, func):
        self.func = func
        self.cache = {}
        self.call_count = 0
        self.hit_count = 0
    def __call__(self, x):
        key = tuple(np.round(x, 6))
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        result = self.func(x)
        self.cache[key] = result
        self.call_count += 1
        return result


class AQPSO:
    """Adaptive Quantum-behaved Particle Swarm Optimization - PARALLEL VERSION."""
    def __init__(self, n_particles=5, max_iter=10, dim=None, bounds=None,
                 alpha=0.6, early_stop_patience=3, min_improvement=1e-5):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.dim = dim
        self.bounds = bounds
        self.alpha = alpha
        self.early_stop_patience = early_stop_patience
        self.min_improvement = min_improvement

    def initialize_particles(self):
        if self.bounds is None:
            self.particles = np.random.randn(self.n_particles, self.dim)
            self.pbest = self.particles.copy()
        else:
            low = np.array([b[0] for b in self.bounds])
            high = np.array([b[1] for b in self.bounds])
            self.particles = np.random.uniform(low, high,
                                                (self.n_particles, self.dim))
            self.pbest = self.particles.copy()
        self.pbest_fitness = np.full(self.n_particles, np.inf)
        self.gbest = self.particles[0].copy()
        self.gbest_fitness = np.inf

    def compute_mbest(self):
        self.mbest = np.mean(self.pbest, axis=0)

    def quantum_update_vectorized(self):
        alpha_t = self.alpha * (1 - self.current_iter / self.max_iter) + 0.5
        u = np.random.random((self.n_particles, self.dim))
        L = alpha_t * np.abs(self.pbest - self.mbest)
        signs = np.where(np.random.random((self.n_particles, self.dim)) > 0.5,
                         1, -1)
        self.particles = self.pbest + signs * L * np.log(1.0 / u)
        if self.bounds is not None:
            low = np.array([b[0] for b in self.bounds])
            high = np.array([b[1] for b in self.bounds])
            np.clip(self.particles, low, high, out=self.particles)

    def optimize(self, objective_func, verbose=True):
        self.initialize_particles()
        no_improve_count = 0
        prev_best = np.inf
        for iteration in range(self.max_iter):
            self.current_iter = iteration
            self.compute_mbest()
            # OPTIMIZATION: Parallel objective evaluation via joblib
            fitness = np.array(Parallel(n_jobs=-1, backend='threading')(
                delayed(objective_func)(p) for p in self.particles
            ))
            improved = fitness < self.pbest_fitness
            self.pbest_fitness[improved] = fitness[improved]
            self.pbest[improved] = self.particles[improved].copy()
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.gbest_fitness:
                self.gbest_fitness = fitness[best_idx]
                self.gbest = self.particles[best_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= self.early_stop_patience:
                if abs(prev_best - self.gbest_fitness) < self.min_improvement:
                    if verbose:
                        print(f"  Early stopping at iteration {iteration+1}")
                    break
                no_improve_count = 0
            prev_best = self.gbest_fitness
            self.quantum_update_vectorized()
            if verbose and (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration+1}/{self.max_iter}, "
                      f"Best Fitness: {self.gbest_fitness:.6f}")
        return self.gbest, self.gbest_fitness


class GWO:
    """Grey Wolf Optimizer - PARALLEL VERSION."""
    def __init__(self, n_wolves=5, max_iter=10, dim=None, bounds=None,
                 early_stop_patience=3, min_improvement=1e-5):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = dim
        self.bounds = bounds
        self.early_stop_patience = early_stop_patience
        self.min_improvement = min_improvement

    def initialize_wolves(self):
        if self.bounds is None:
            self.wolves = np.random.randn(self.n_wolves, self.dim)
        else:
            low = np.array([b[0] for b in self.bounds])
            high = np.array([b[1] for b in self.bounds])
            self.wolves = np.random.uniform(low, high,
                                            (self.n_wolves, self.dim))
        self.alpha_pos = self.wolves[0].copy()
        self.alpha_score = np.inf
        self.beta_pos = (self.wolves[1].copy() if self.n_wolves > 1
                         else self.wolves[0].copy())
        self.beta_score = np.inf
        self.delta_pos = (self.wolves[2].copy() if self.n_wolves > 2
                          else self.wolves[0].copy())
        self.delta_score = np.inf

    def update_hierarchy(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        if fitness_values[sorted_indices[0]] < self.alpha_score:
            self.alpha_score = fitness_values[sorted_indices[0]]
            self.alpha_pos = self.wolves[sorted_indices[0]].copy()
        if (len(sorted_indices) > 1 and
                fitness_values[sorted_indices[1]] < self.beta_score):
            self.beta_score = fitness_values[sorted_indices[1]]
            self.beta_pos = self.wolves[sorted_indices[1]].copy()
        if (len(sorted_indices) > 2 and
                fitness_values[sorted_indices[2]] < self.delta_score):
            self.delta_score = fitness_values[sorted_indices[2]]
            self.delta_pos = self.wolves[sorted_indices[2]].copy()

    def hunt_vectorized(self, a):
        r1, r2, r3, r4, r5, r6 = [np.random.random((self.n_wolves, self.dim))
                                    for _ in range(6)]
        A1 = 2 * a * r1 - a; C1 = 2 * r2
        A2 = 2 * a * r3 - a; C2 = 2 * r4
        A3 = 2 * a * r5 - a; C3 = 2 * r6
        D_alpha = np.abs(C1 * self.alpha_pos - self.wolves)
        D_beta  = np.abs(C2 * self.beta_pos - self.wolves)
        D_delta = np.abs(C3 * self.delta_pos - self.wolves)
        X1 = self.alpha_pos - A1 * D_alpha
        X2 = self.beta_pos  - A2 * D_beta
        X3 = self.delta_pos - A3 * D_delta
        self.wolves = (X1 + X2 + X3) / 3.0
        if self.bounds is not None:
            low = np.array([b[0] for b in self.bounds])
            high = np.array([b[1] for b in self.bounds])
            np.clip(self.wolves, low, high, out=self.wolves)

    def optimize(self, objective_func, verbose=True):
        self.initialize_wolves()
        # OPTIMIZATION: Parallel initial fitness evaluation
        fitness_values = np.array(Parallel(n_jobs=-1, backend='threading')(
            delayed(objective_func)(w) for w in self.wolves
        ))
        self.update_hierarchy(fitness_values)
        no_improve_count = 0
        prev_best = np.inf
        for iteration in range(self.max_iter):
            a = 2 - iteration * (2.0 / self.max_iter)
            self.hunt_vectorized(a)
            # OPTIMIZATION: Parallel fitness evaluation each iteration
            fitness_values = np.array(Parallel(n_jobs=-1, backend='threading')(
                delayed(objective_func)(w) for w in self.wolves
            ))
            self.update_hierarchy(fitness_values)
            if self.alpha_score < prev_best - self.min_improvement:
                no_improve_count = 0
                prev_best = self.alpha_score
            else:
                no_improve_count += 1
            if no_improve_count >= self.early_stop_patience:
                if verbose:
                    print(f"  Early stopping at iteration {iteration+1}")
                break
            if verbose and (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration+1}/{self.max_iter}, "
                      f"Alpha Score: {self.alpha_score:.6f}")
        return self.alpha_pos, self.alpha_score


class HybridAQPSOGWO:
    """Hybrid AQPSO-GWO Optimizer - OPTIMIZED VERSION."""
    def __init__(self, n_agents=5, max_iter=10, dim=None, bounds=None,
                 alpha=0.6, early_stop_patience=3):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.dim = dim
        self.bounds = bounds
        self.alpha = alpha
        self.early_stop_patience = early_stop_patience

    def initialize_agents(self):
        if self.bounds is None:
            self.agents = np.random.randn(self.n_agents, self.dim)
        else:
            low = np.array([b[0] for b in self.bounds])
            high = np.array([b[1] for b in self.bounds])
            self.agents = np.random.uniform(low, high,
                                            (self.n_agents, self.dim))
        self.aqpso = AQPSO(n_particles=self.n_agents, max_iter=self.max_iter,
                           dim=self.dim, bounds=self.bounds,
                           alpha=self.alpha,
                           early_stop_patience=self.early_stop_patience)
        self.gwo = GWO(n_wolves=self.n_agents, max_iter=self.max_iter,
                       dim=self.dim, bounds=self.bounds,
                       early_stop_patience=self.early_stop_patience)
        self.aqpso.particles = self.agents.copy()
        self.aqpso.pbest = self.agents.copy()
        self.aqpso.pbest_fitness = np.full(self.n_agents, np.inf)
        self.gwo.wolves = self.agents.copy()

    def _aqpso_update_single(self, i):
        particle = self.aqpso.particles[i].copy()
        pbest = self.aqpso.pbest[i].copy()
        mbest = self.aqpso.mbest.copy()
        alpha_t = (self.aqpso.alpha *
                   (1 - self.current_iter / self.aqpso.max_iter) + 0.5)
        for d in range(self.dim):
            u = np.random.random()
            L = alpha_t * abs(pbest[d] - mbest[d])
            particle[d] = (pbest[d] +
                           (1 if np.random.random() > 0.5 else -1) *
                           L * np.log(1.0 / u))
            if self.bounds is not None:
                particle[d] = np.clip(particle[d], self.bounds[d][0],
                                      self.bounds[d][1])
        return particle

    def _gwo_hunt_single(self, i, a):
        wolf = self.gwo.wolves[i].copy()
        for d in range(self.dim):
            r1, r2, r3, r4, r5, r6 = np.random.random(6)
            A1, C1 = 2*a*r1 - a, 2*r2
            A2, C2 = 2*a*r3 - a, 2*r4
            A3, C3 = 2*a*r5 - a, 2*r6
            D_alpha = abs(C1 * self.gwo.alpha_pos[d] - wolf[d])
            D_beta  = abs(C2 * self.gwo.beta_pos[d] - wolf[d])
            D_delta = abs(C3 * self.gwo.delta_pos[d] - wolf[d])
            wolf[d] = ((self.gwo.alpha_pos[d] - A1*D_alpha +
                        self.gwo.beta_pos[d] - A2*D_beta +
                        self.gwo.delta_pos[d] - A3*D_delta) / 3.0)
            if self.bounds is not None:
                wolf[d] = np.clip(wolf[d], self.bounds[d][0],
                                  self.bounds[d][1])
        return wolf

    def optimize(self, objective_func, verbose=True):
        self.initialize_agents()
        # OPTIMIZATION: Parallel initial fitness evaluation
        fitness = np.array(Parallel(n_jobs=-1, backend='threading')(
            delayed(objective_func)(agent) for agent in self.agents
        ))
        best_idx = np.argmin(fitness)
        self.best_pos = self.agents[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.aqpso.gbest = self.best_pos.copy()
        self.aqpso.gbest_fitness = self.best_fitness
        self.gwo.alpha_pos = self.best_pos.copy()
        self.gwo.alpha_score = self.best_fitness
        sorted_indices = np.argsort(fitness)
        if len(sorted_indices) > 1:
            self.gwo.beta_pos = self.agents[sorted_indices[1]].copy()
            self.gwo.beta_score = fitness[sorted_indices[1]]
        if len(sorted_indices) > 2:
            self.gwo.delta_pos = self.agents[sorted_indices[2]].copy()
            self.gwo.delta_score = fitness[sorted_indices[2]]

        no_improve_count = 0
        prev_best = np.inf
        for iteration in range(self.max_iter):
            self.current_iter = iteration
            current_ratio = (0.5 + 0.4 *
                             np.cos(np.pi * iteration / self.max_iter))
            n_aqpso = max(1, int(self.n_agents * current_ratio))
            n_gwo = self.n_agents - n_aqpso

            if n_aqpso > 0:
                self.aqpso.current_iter = iteration
                self.aqpso.compute_mbest()
                # OPTIMIZATION: Parallel AQPSO agent updates
                aqpso_agents = Parallel(n_jobs=-1, backend='threading')(
                    delayed(self._aqpso_update_single)(i)
                    for i in range(n_aqpso)
                )
                for i in range(n_aqpso):
                    self.agents[i] = aqpso_agents[i]
                    fitness_val = objective_func(self.agents[i])
                    if fitness_val < self.aqpso.pbest_fitness[i]:
                        self.aqpso.pbest_fitness[i] = fitness_val
                        self.aqpso.pbest[i] = self.agents[i].copy()
                    if fitness_val < self.best_fitness:
                        self.best_fitness = fitness_val
                        self.best_pos = self.agents[i].copy()

            if n_gwo > 0:
                a = 2 - iteration * (2.0 / self.max_iter)
                # OPTIMIZATION: Parallel GWO agent updates
                gwo_agents = Parallel(n_jobs=-1, backend='threading')(
                    delayed(self._gwo_hunt_single)(i, a)
                    for i in range(n_aqpso, self.n_agents)
                )
                for i, idx in enumerate(range(n_aqpso, self.n_agents)):
                    self.agents[idx] = gwo_agents[i]
                    fitness_val = objective_func(self.agents[idx])
                    if fitness_val < self.best_fitness:
                        self.best_fitness = fitness_val
                        self.best_pos = self.agents[idx].copy()

            # OPTIMIZATION: Parallel final fitness for hierarchy update
            all_fitness = np.array(Parallel(n_jobs=-1, backend='threading')(
                delayed(objective_func)(agent) for agent in self.agents
            ))
            self.gwo.wolves = self.agents.copy()
            self.gwo.update_hierarchy(all_fitness)
            self.aqpso.particles = self.agents.copy()

            if self.gwo.alpha_score < self.best_fitness:
                self.best_fitness = self.gwo.alpha_score
                self.best_pos = self.gwo.alpha_pos.copy()
            if self.aqpso.gbest_fitness < self.best_fitness:
                self.best_fitness = self.aqpso.gbest_fitness
                self.best_pos = self.aqpso.gbest.copy()

            if self.best_fitness < prev_best - 1e-5:
                no_improve_count = 0
                prev_best = self.best_fitness
            else:
                no_improve_count += 1
            if no_improve_count >= self.early_stop_patience:
                if verbose:
                    print(f"  Early stopping at iteration {iteration+1}")
                break
            if verbose and (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration+1}/{self.max_iter}, "
                      f"Best: {self.best_fitness:.6f}, "
                      f"Ratio: {current_ratio:.2f}")
        return self.best_pos, self.best_fitness

print("Optimizers defined (PARALLEL evaluation enabled)")


# ==============================================================================
# CELL 3: PREPROCESSING
# ==============================================================================
print("\n" + "="*70)
print("PREPROCESSING PIPELINE (OPTIMIZED)")
print("="*70)


def load_cleaned_data(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    print(f"Found {len(csv_files)} CSV file(s) in: {folder_path}")
    if not csv_files:
        raise ValueError(f"No CSV files found in: {folder_path}")
    dataframes = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if WELL_COL not in df.columns:
                well_name = Path(f).stem.replace('_CLEANED', '').replace('_cleaned', '')
                df[WELL_COL] = well_name
            if DEPTH_COL in df.columns:
                df[DEPTH_COL] = pd.to_numeric(df[DEPTH_COL], errors='coerce')
            dataframes.append(df)
            print(f"  Loaded: {Path(f).name} ({len(df)} rows, {len(df.columns)} cols)")
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    if not dataframes:
        raise ValueError("No data loaded!")
    full_df = pd.concat(dataframes, ignore_index=True, copy=False)
    full_df[WELL_COL] = full_df[WELL_COL].astype('category')
    full_df = full_df.sort_values([WELL_COL, DEPTH_COL]).reset_index(drop=True)
    print(f"\nCombined dataset: {full_df.shape[0]} samples from "
          f"{full_df[WELL_COL].nunique()} wells")
    print(f"  Wells: {sorted(full_df[WELL_COL].unique())}")
    return full_df


def drop_sparse_features(df, threshold=0.7):
    missing_ratio = df.isnull().mean()
    to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    protected = [WELL_COL, DEPTH_COL, TARGET_COLUMN]
    to_drop = [c for c in to_drop if c not in protected]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"Dropped sparse features (> {threshold*100:.0f}% NaN): {to_drop}")
    else:
        print(f"No sparse features to drop (threshold = {threshold*100:.0f}%)")
    return df


def minimal_feature_engineering(df):
    df = df.copy()
    if DEPTH_COL in df.columns:
        well_groups = df.groupby(WELL_COL)[DEPTH_COL]
        depth_min = well_groups.transform('min')
        depth_max = well_groups.transform('max')
        df['REL_DEPTH'] = ((df[DEPTH_COL] - depth_min) /
                           (depth_max - depth_min + 1e-6))
        print("Added REL_DEPTH (compaction trend)")
    return df


def handle_outliers_by_well(df, features):
    """Winsorize outliers per well using vectorized groupby."""
    df_clean = df.copy()
    for col in features:
        if col in df_clean.columns:
            lo = df_clean.groupby(WELL_COL)[col].transform(
                lambda x: x.quantile(0.01))
            hi = df_clean.groupby(WELL_COL)[col].transform(
                lambda x: x.quantile(0.99))
            df_clean[col] = df_clean[col].clip(lower=lo, upper=hi)
    print("Winsorized outliers per well (1%-99%) [vectorized]")
    return df_clean.reset_index(drop=True)


def depth_aware_imputation(df, features):
    """Linear interpolation by DEPTH - vectorized per-well imputation."""
    df_imputed = df.copy()
    total_wells = df_imputed[WELL_COL].nunique()
    print(f"  Starting per-well imputation for {total_wells} wells...")
    for well_idx, well in enumerate(df_imputed[WELL_COL].unique()):
        mask = df_imputed[WELL_COL] == well
        well_df = df_imputed.loc[mask].sort_values(DEPTH_COL).copy()
        for col in features:
            if col not in well_df.columns:
                continue
            if well_df[col].isnull().sum() == 0:
                continue
            temp = well_df.set_index(DEPTH_COL)[col].sort_index()
            temp = temp.interpolate(method='linear',
                                    limit_direction='both').ffill().bfill()
            well_df[col] = temp.reindex(well_df[DEPTH_COL]).values
            if well_df[col].isnull().sum() > 0:
                med = well_df[col].median()
                if pd.notna(med):
                    well_df[col] = well_df[col].fillna(med)
        df_imputed.loc[mask, features] = well_df[features].values
        if (well_idx + 1) % 3 == 0 or well_idx == total_wells - 1:
            print(f"    ...processed {well_idx + 1}/{total_wells} wells",
                  end='\r')
    print()
    for col in features:
        if col in df_imputed.columns and df_imputed[col].isnull().sum() > 0:
            gmed = df_imputed[col].median()
            if pd.notna(gmed):
                df_imputed[col] = df_imputed[col].fillna(gmed)
    print("Depth-aware imputation complete (memory-safe)")
    return df_imputed


def remove_highly_correlated_features(df, features, threshold=0.95):
    if len(features) < 2:
        return df, features
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns
               if any(upper[column] > threshold)]
    if to_drop:
        print(f"Dropped {len(to_drop)} highly correlated features "
              f"(|r| > {threshold}): {to_drop}")
        df = df.drop(columns=to_drop)
        features = [f for f in features if f not in to_drop]
    else:
        print(f"No highly correlated pairs (threshold = {threshold})")
    return df, features


def full_preprocessing(folder_path):
    df = load_cleaned_data(folder_path)
    exclude = [WELL_COL, DEPTH_COL, TARGET_COLUMN]
    features = [c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    print(f"\nInitial numeric features ({len(features)}): {features}")
    df = drop_sparse_features(df, threshold=0.7)
    features = [c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    df = handle_outliers_by_well(df, features)
    df = depth_aware_imputation(df, features)
    df = minimal_feature_engineering(df)
    features = [c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    df, features = remove_highly_correlated_features(df, features,
                                                      threshold=0.95)
    # OPTIMIZATION: Skipped per-well normalization - global RobustScaler handles it
    print("Skipped per-well normalization (using global RobustScaler after split)")
    df = df.dropna(subset=features + [TARGET_COLUMN]).reset_index(drop=True)
    final_path = os.path.join(run_dir, 'ALL_combined_final.csv')
    save_cols = [WELL_COL, DEPTH_COL] + features + [TARGET_COLUMN]
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_csv(final_path, index=False)
    print(f"\nSaved final combined: {final_path} ({len(df)} rows)")
    print(f"\n{'='*60}")
    print(f"FINAL DATASET: {df.shape[0]} samples, {len(features)} features")
    print(f"Wells: {df[WELL_COL].nunique()}")
    print(f"Target range: {df[TARGET_COLUMN].min():.4f} - "
          f"{df[TARGET_COLUMN].max():.4f}")
    print(f"Features: {features}")
    print(f"{'='*60}")
    return df, features


df_processed, feature_cols = full_preprocessing(INPUT_PATH)

# ==============================================================================
# CELL 4: RANDOM TRAIN/TEST SPLIT
# ==============================================================================
print("\n" + "="*70)
print("RANDOM TRAIN/TEST SPLIT (80/20 across ALL data)")
print("="*70)

X = df_processed[feature_cols].values.astype(np.float32)
y = df_processed[TARGET_COLUMN].values.astype(np.float32)
wells = df_processed[WELL_COL].values

print(f"Total samples: {len(X)}")
print(f"Features: {len(feature_cols)}")
print(f"Feature list: {feature_cols}")

X_train, X_test, y_train, y_test, wells_train, wells_test = train_test_split(
    X, y, wells, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
)

print(f"\nTRAIN-TEST SPLIT (random across all wells):")
print(f"  Train samples: {X_train.shape[0]} "
      f"({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test samples:  {X_test.shape[0]} "
      f"({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"  Wells in train: {len(np.unique(wells_train))}")
print(f"  Wells in test:  {len(np.unique(wells_test))}")

scaler_X = RobustScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
scaler_y = RobustScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

with open(os.path.join(models_dir, 'scaler_X.pkl'), 'wb') as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(models_dir, 'scaler_y.pkl'), 'wb') as f:
    pickle.dump(scaler_y, f)

input_dim = X_train_scaled.shape[1]
print(f"\nRobustScaler fitted on training data (global normalization)")
print(f"Input dimension: {input_dim}")


# ==============================================================================
# CELL 5: ML MODEL CONFIGURATIONS (GPU-ACCELERATED)
# ==============================================================================
# OPTIMIZATION: Full GPU parameters for XGBoost
xgb_fixed = {
    'random_state': RANDOM_SEED, 'n_jobs': -1, 'n_estimators': 100,
    'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'colsample_bytree': 0.8,
    'tree_method': 'hist',
}
# XGBoost: use 'hist' (fast CPU histogram) - 'gpu_hist' requires GPU-enabled XGBoost build
xgb_fixed['tree_method'] = 'hist'
print("XGBoost: tree_method='hist' (fast CPU histogram algorithm)")

# OPTIMIZATION: GPU parameters for LightGBM
lgbm_fixed = {
    'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbose': -1,
    'n_estimators': 100
}
if torch.cuda.is_available():
    lgbm_fixed['device'] = 'gpu'
    lgbm_fixed['gpu_platform_id'] = 0
    lgbm_fixed['gpu_device_id'] = 0
    print("LightGBM: GPU mode enabled")

# OPTIMIZATION: Use cuML GPU models when available
RF_ModelClass = cuRF if (CUML_AVAILABLE and torch.cuda.is_available()) else RandomForestRegressor
ET_ModelClass = cuET if (CUML_AVAILABLE and torch.cuda.is_available()) else ExtraTreesRegressor
if CUML_AVAILABLE and torch.cuda.is_available():
    print("RandomForest: GPU (cuML)")
    print("ExtraTrees: GPU (cuML)")

ML_MODEL_CONFIGS = {
    'RandomForest': {
        'model': RF_ModelClass,
        'params': ['n_estimators', 'max_depth', 'min_samples_split',
                   'min_samples_leaf'],
        'bounds': [(50, 300), (5, 30), (2, 20), (1, 10)],
        'int_params': [0, 1, 2, 3],
        'fixed': {'n_jobs': -1, 'random_state': RANDOM_SEED}
    },
    'XGBoost': {
        'model': XGBRegressor if XGBOOST_AVAILABLE else None,
        'params': ['n_estimators', 'max_depth', 'learning_rate', 'subsample'],
        'bounds': [(50, 300), (3, 12), (0.01, 0.3), (0.6, 1.0)],
        'int_params': [0, 1],
        'float_params': [2, 3],
        'fixed': xgb_fixed
    },
    'LightGBM': {
        'model': LGBMRegressor if LIGHTGBM_AVAILABLE else None,
        'params': ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves'],
        'bounds': [(50, 300), (3, 15), (0.01, 0.3), (20, 150)],
        'int_params': [0, 1, 3],
        'float_params': [2],
        'fixed': lgbm_fixed
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor,
        'params': ['n_estimators', 'max_depth', 'learning_rate', 'subsample'],
        'bounds': [(50, 300), (3, 12), (0.01, 0.3), (0.6, 1.0)],
        'int_params': [0, 1],
        'float_params': [2, 3],
        'fixed': {'random_state': RANDOM_SEED}
    },
    'ExtraTrees': {
        'model': ET_ModelClass,
        'params': ['n_estimators', 'max_depth', 'min_samples_split',
                   'min_samples_leaf'],
        'bounds': [(50, 300), (5, 30), (2, 20), (1, 10)],
        'int_params': [0, 1, 2, 3],
        'fixed': {'n_jobs': -1, 'random_state': RANDOM_SEED}
    },
    'AdaBoost': {
        'model': AdaBoostRegressor,
        'params': ['n_estimators', 'learning_rate'],
        'bounds': [(50, 300), (0.01, 1.0)],
        'int_params': [0],
        'float_params': [1],
        'fixed': {'random_state': RANDOM_SEED}
    }
}

print("ML model configurations defined (GPU-accelerated)")

# ==============================================================================
# CELL 6: PYTORCH DL MODELS (GPU-OPTIMIZED)
# ==============================================================================
class WellLogDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units[0], hidden_units[1]), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units[1], 1)
        )
    def forward(self, x):
        return self.network(x)


class OptimizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate, l2_reg):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, units in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)


class ResidualDNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.hidden = nn.ModuleList()
        self.residual_flags = []
        for i in range(1, len(hidden_layers)):
            self.hidden.append(nn.Sequential(
                nn.Linear(hidden_layers[i-1], hidden_layers[i]),
                nn.BatchNorm1d(hidden_layers[i]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            self.residual_flags.append(hidden_layers[i-1] == hidden_layers[i])
        self.output = nn.Linear(hidden_layers[-1], 1)
    def forward(self, x):
        x = self.input_layer(x)
        for block, use_residual in zip(self.hidden, self.residual_flags):
            if use_residual:
                x = block(x) + x
            else:
                x = block(x)
        return self.output(x)


class PINN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)


def train_pytorch_model(model, train_loader, val_loader, epochs=50, lr=0.001,
                        patience=10, device='cpu', weight_decay=1e-3):
    """Train PyTorch model with GPU optimizations (AMP, compile, set_to_none)."""
    model = model.to(device)
    # NOTE: torch.compile() removed - requires Triton (not installed)
    # If Triton is available, uncomment: model = torch.compile(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    # OPTIMIZATION: Reduced scheduler patience for faster convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)
    # OPTIMIZATION: Automatic Mixed Precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            # OPTIMIZATION: non_blocking for async GPU transfer
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            # OPTIMIZATION: set_to_none=True is faster than zero_grad()
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                # OPTIMIZATION: AMP autocast for faster forward pass
                with torch.cuda.amp.autocast():
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

        # Validation with AMP
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                if scaler:
                    with torch.cuda.amp.autocast():
                        pred = model(X_batch)
                        val_loss += criterion(pred, y_batch).item()
                else:
                    pred = model(X_batch)
                    val_loss += criterion(pred, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def predict_in_batches(model, X, device, batch_size=1024):
    """Predict in batches to avoid GPU OOM."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(
                device, non_blocking=True)
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    pred = model(batch).cpu().numpy()
            else:
                pred = model(batch).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds).ravel()


DL_MODEL_CONFIGS = {
    'SimpleMLP': {
        'class': SimpleMLP,
        'params': ['hidden_units_1', 'hidden_units_2', 'dropout_rate',
                   'learning_rate'],
        'bounds': [(32, 256), (16, 128), (0.1, 0.5), (0.0001, 0.01)],
        'int_params': [0, 1]
    },
    'OptimizedMLP': {
        'class': OptimizedMLP,
        'params': ['hidden_units_1', 'hidden_units_2', 'hidden_units_3',
                   'dropout_rate', 'l2_reg', 'learning_rate'],
        'bounds': [(32, 256), (32, 128), (16, 64), (0.1, 0.5),
                   (0.0001, 0.01), (0.0001, 0.01)],
        'int_params': [0, 1, 2]
    },
    'ResidualDNN': {
        'class': ResidualDNN,
        'params': ['hidden_units_1', 'hidden_units_2', 'hidden_units_3',
                   'dropout_rate', 'learning_rate'],
        'bounds': [(32, 256), (32, 256), (16, 128), (0.1, 0.5), (0.0001, 0.01)],
        'int_params': [0, 1, 2]
    },
    'PINN': {
        'class': PINN,
        'params': ['hidden_units_1', 'hidden_units_2', 'hidden_units_3',
                   'dropout_rate', 'learning_rate'],
        'bounds': [(32, 256), (32, 256), (16, 128), (0.1, 0.5), (0.0001, 0.01)],
        'int_params': [0, 1, 2]
    }
}

print("PyTorch DL models defined (GPU-optimized with AMP)")


# ==============================================================================
# CELL 7: OBJECTIVE FUNCTIONS (OPTIMIZED - 3-fold CV)
# ==============================================================================
def create_ml_objective(model_name, X_train, y_train, cv_folds=3):
    """Create ML objective function with OPTIMIZED 3-fold CV."""
    config = ML_MODEL_CONFIGS[model_name]
    model_class = config['model']
    if model_class is None:
        return lambda x: np.inf
    params = config['params']
    int_params = config.get('int_params', [])
    fixed = config.get('fixed', {}).copy()

    def objective(hyperparams):
        param_dict = fixed.copy()
        for i, param in enumerate(params):
            if i in int_params:
                param_dict[param] = int(round(hyperparams[i]))
            else:
                param_dict[param] = hyperparams[i]
        try:
            kfold = KFold(n_splits=cv_folds, shuffle=True,
                          random_state=RANDOM_SEED)
            scores = []
            for tr_idx, val_idx in kfold.split(X_train):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                model = model_class(**param_dict)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                scores.append(r2_score(y_val, pred))
            return -np.mean(scores)
        except Exception:
            return np.inf
    return CachedObjective(objective)


def create_dl_objective(model_name, X_train, y_train, epochs=10):
    """Create DL objective function with OPTIMIZED 3-fold CV."""
    config = DL_MODEL_CONFIGS[model_name]
    model_class = config['class']
    params = config['params']
    int_params = config.get('int_params', [])
    input_dim_local = X_train.shape[1]

    def objective(hyperparams):
        param_dict = {}
        for i, param in enumerate(params):
            if i in int_params:
                param_dict[param] = int(round(hyperparams[i]))
            else:
                param_dict[param] = hyperparams[i]
        try:
            kfold = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            val_losses = []
            for tr_idx, val_idx in kfold.split(X_train):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                if model_name == 'SimpleMLP':
                    model = model_class(
                        input_dim=input_dim_local,
                        hidden_units=[param_dict['hidden_units_1'],
                                      param_dict['hidden_units_2']],
                        dropout_rate=param_dict['dropout_rate'])
                elif model_name == 'OptimizedMLP':
                    model = model_class(
                        input_dim=input_dim_local,
                        hidden_layers=[param_dict['hidden_units_1'],
                                       param_dict['hidden_units_2'],
                                       param_dict['hidden_units_3']],
                        dropout_rate=param_dict['dropout_rate'],
                        l2_reg=param_dict['l2_reg'])
                else:
                    model = model_class(
                        input_dim=input_dim_local,
                        hidden_layers=[param_dict['hidden_units_1'],
                                       param_dict['hidden_units_2'],
                                       param_dict['hidden_units_3']],
                        dropout_rate=param_dict['dropout_rate'])
                train_ds = WellLogDataset(X_tr, y_tr)
                val_ds = WellLogDataset(X_val, y_val)
                # OPTIMIZATION: batch_size=256
                train_loader = DataLoader(
                    train_ds, batch_size=dl_batch_size, shuffle=False,
                    drop_last=True, num_workers=0)
                val_loader = DataLoader(
                    val_ds, batch_size=dl_batch_size, shuffle=False,
                    num_workers=0)
                _, val_loss = train_pytorch_model(
                    model, train_loader, val_loader,
                    epochs=epochs, lr=param_dict['learning_rate'],
                    patience=5, device=device, weight_decay=1e-3)
                val_losses.append(val_loss)
            return np.mean(val_losses)
        except Exception:
            return np.inf
    return CachedObjective(objective)

print("Objective functions defined (3-fold CV, batch_size=256)")

# ==============================================================================
# CELLS 8: BASELINE ML MODELS
# ==============================================================================
print("\n" + "="*70)
print("BASELINE ML MODELS (Random Split)")
print("="*70)

baseline_results = {}

print("\nBASELINE: Random Forest")
start_time = time.time()
rf_baseline = RF_ModelClass(
    random_state=RANDOM_SEED, n_jobs=-1,
    n_estimators=100, max_depth=20, min_samples_leaf=4)
rf_baseline.fit(X_train_scaled, y_train_scaled)
y_pred_train = scaler_y.inverse_transform(
    rf_baseline.predict(X_train_scaled).reshape(-1, 1)).ravel()
y_pred_test = scaler_y.inverse_transform(
    rf_baseline.predict(X_test_scaled).reshape(-1, 1)).ravel()
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
print(f"Train R: {train_r2:.6f} | Test R: {test_r2:.6f} | "
      f"RMSE: {rmse:.6f} | Time: {time.time()-start_time:.1f}s")
baseline_results['Random Forest'] = {
    'model': rf_baseline, 'train_r2': train_r2, 'test_r2': test_r2,
    'rmse': rmse, 'mae': mae, 'y_pred_test': y_pred_test}
save_checkpoint()

if XGBOOST_AVAILABLE:
    print("\nBASELINE: XGBoost")
    start_time = time.time()
    xgb_baseline = XGBRegressor(**xgb_fixed)
    xgb_baseline.fit(X_train_scaled, y_train_scaled)
    y_pred_train = scaler_y.inverse_transform(
        xgb_baseline.predict(X_train_scaled).reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(
        xgb_baseline.predict(X_test_scaled).reshape(-1, 1)).ravel()
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Train R: {train_r2:.6f} | Test R: {test_r2:.6f} | "
          f"RMSE: {rmse:.6f} | Time: {time.time()-start_time:.1f}s")
    baseline_results['XGBoost'] = {
        'model': xgb_baseline, 'train_r2': train_r2, 'test_r2': test_r2,
        'rmse': rmse, 'mae': mae, 'y_pred_test': y_pred_test}
    save_checkpoint()

if LIGHTGBM_AVAILABLE:
    print("\nBASELINE: LightGBM")
    start_time = time.time()
    lgbm_baseline = LGBMRegressor(**lgbm_fixed)
    lgbm_baseline.fit(X_train_scaled, y_train_scaled)
    y_pred_train = scaler_y.inverse_transform(
        lgbm_baseline.predict(X_train_scaled).reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(
        lgbm_baseline.predict(X_test_scaled).reshape(-1, 1)).ravel()
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Train R: {train_r2:.6f} | Test R: {test_r2:.6f} | "
          f"RMSE: {rmse:.6f} | Time: {time.time()-start_time:.1f}s")
    baseline_results['LightGBM'] = {
        'model': lgbm_baseline, 'train_r2': train_r2, 'test_r2': test_r2,
        'rmse': rmse, 'mae': mae, 'y_pred_test': y_pred_test}
    save_checkpoint()

print("\nBASELINE: Gradient Boosting")
start_time = time.time()
gb_baseline = GradientBoostingRegressor(random_state=RANDOM_SEED,
                                        n_estimators=100)
gb_baseline.fit(X_train_scaled, y_train_scaled)
y_pred_train = scaler_y.inverse_transform(
    gb_baseline.predict(X_train_scaled).reshape(-1, 1)).ravel()
y_pred_test = scaler_y.inverse_transform(
    gb_baseline.predict(X_test_scaled).reshape(-1, 1)).ravel()
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
print(f"Train R: {train_r2:.6f} | Test R: {test_r2:.6f} | "
      f"RMSE: {rmse:.6f} | Time: {time.time()-start_time:.1f}s")
baseline_results['Gradient Boosting'] = {
    'model': gb_baseline, 'train_r2': train_r2, 'test_r2': test_r2,
    'rmse': rmse, 'mae': mae, 'y_pred_test': y_pred_test}
save_checkpoint()

print("\nBASELINE: Extra Trees")
start_time = time.time()
et_baseline = ET_ModelClass(
    random_state=RANDOM_SEED, n_jobs=-1, n_estimators=100, max_depth=20)
et_baseline.fit(X_train_scaled, y_train_scaled)
y_pred_train = scaler_y.inverse_transform(
    et_baseline.predict(X_train_scaled).reshape(-1, 1)).ravel()
y_pred_test = scaler_y.inverse_transform(
    et_baseline.predict(X_test_scaled).reshape(-1, 1)).ravel()
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
print(f"Train R: {train_r2:.6f} | Test R: {test_r2:.6f} | "
      f"RMSE: {rmse:.6f} | Time: {time.time()-start_time:.1f}s")
baseline_results['Extra Trees'] = {
    'model': et_baseline, 'train_r2': train_r2, 'test_r2': test_r2,
    'rmse': rmse, 'mae': mae, 'y_pred_test': y_pred_test}
save_checkpoint()

print("\nBASELINE: AdaBoost")
start_time = time.time()
ada_baseline = AdaBoostRegressor(random_state=RANDOM_SEED, n_estimators=100)
ada_baseline.fit(X_train_scaled, y_train_scaled)
y_pred_train = scaler_y.inverse_transform(
    ada_baseline.predict(X_train_scaled).reshape(-1, 1)).ravel()
y_pred_test = scaler_y.inverse_transform(
    ada_baseline.predict(X_test_scaled).reshape(-1, 1)).ravel()
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
print(f"Train R: {train_r2:.6f} | Test R: {test_r2:.6f} | "
      f"RMSE: {rmse:.6f} | Time: {time.time()-start_time:.1f}s")
baseline_results['AdaBoost'] = {
    'model': ada_baseline, 'train_r2': train_r2, 'test_r2': test_r2,
    'rmse': rmse, 'mae': mae, 'y_pred_test': y_pred_test}
save_checkpoint()

print("\n" + "="*70)
print("BASELINE ML MODELS SUMMARY")
print("="*70)
baseline_ml_summary = []
for name, result in baseline_results.items():
    if name in ['Random Forest', 'XGBoost', 'LightGBM',
                'Gradient Boosting', 'Extra Trees', 'AdaBoost']:
        baseline_ml_summary.append({
            'Model': name, 'Train R2': result['train_r2'],
            'Test R2': result['test_r2'], 'RMSE': result['rmse'],
            'MAE': result['mae']})
summary_df = pd.DataFrame(baseline_ml_summary).sort_values('Test R2',
                                                              ascending=False)
print("\n" + summary_df.to_string(index=False))


# ==============================================================================
# CELLS 9: BASELINE DL MODELS (OPTIMIZED - 20 epochs, batch_size=256)
# ==============================================================================
print("\n" + "="*70)
print("BASELINE DL MODELS (PyTorch - Random Split)")
print("="*70)


def evaluate_pytorch_baseline(model_class, model_name, hidden_config,
                               dropout=0.3, lr=0.001, epochs=20):
    """Evaluate a PyTorch baseline model with GPU optimizations."""
    print(f"\nBASELINE: {model_name}")
    start_time = time.time()
    if model_name == 'Simple MLP':
        model = model_class(input_dim=input_dim, hidden_units=hidden_config,
                            dropout_rate=dropout)
    elif model_name == 'Optimized MLP':
        model = model_class(input_dim=input_dim, hidden_layers=hidden_config,
                            dropout_rate=dropout, l2_reg=1e-3)
    else:
        model = model_class(input_dim=input_dim, hidden_layers=hidden_config,
                            dropout_rate=dropout)

    train_ds = WellLogDataset(X_train_scaled, y_train_scaled)
    val_ds = WellLogDataset(X_test_scaled, y_test_scaled)
    # OPTIMIZATION: batch_size=256
    train_loader = DataLoader(
        train_ds, batch_size=dl_batch_size, shuffle=False,
        drop_last=True, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=dl_batch_size, shuffle=False, num_workers=0)
    model, _ = train_pytorch_model(
        model, train_loader, val_loader, epochs=epochs, lr=lr,
        patience=10, device=device, weight_decay=1e-3)
    # OPTIMIZATION: Batch prediction instead of full-tensor (avoids GPU OOM)
    train_pred = predict_in_batches(model, X_train_scaled, device)
    test_pred = predict_in_batches(model, X_test_scaled, device)

    y_pred_train = scaler_y.inverse_transform(
        train_pred.reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(
        test_pred.reshape(-1, 1)).ravel()
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Train R: {train_r2:.4f} | Test R: {test_r2:.4f} | "
          f"RMSE: {rmse:.4f} | Time: {time.time()-start_time:.1f}s")
    return {
        'model': model, 'train_r2': train_r2, 'test_r2': test_r2,
        'rmse': rmse, 'mae': mae, 'y_pred_test': y_pred_test}


# OPTIMIZATION: Reduced epochs from 30 to 20
baseline_results['Simple MLP'] = evaluate_pytorch_baseline(
    SimpleMLP, 'Simple MLP', [64, 32], dropout=0.3, lr=0.001,
    epochs=dl_base_epochs)
save_checkpoint()
baseline_results['Optimized MLP'] = evaluate_pytorch_baseline(
    OptimizedMLP, 'Optimized MLP', [64, 32, 16], dropout=0.3, lr=0.001,
    epochs=dl_base_epochs)
save_checkpoint()
baseline_results['Residual DNN'] = evaluate_pytorch_baseline(
    ResidualDNN, 'Residual DNN', [64, 64, 32], dropout=0.3, lr=0.001,
    epochs=dl_base_epochs)
save_checkpoint()
baseline_results['PINN'] = evaluate_pytorch_baseline(
    PINN, 'PINN', [64, 64, 32], dropout=0.3, lr=0.001,
    epochs=dl_base_epochs)
save_checkpoint()

print("\n" + "="*70)
print("BASELINE DL MODELS SUMMARY")
print("="*70)
baseline_dl_summary = []
for name in ['Simple MLP', 'Optimized MLP', 'Residual DNN', 'PINN']:
    if name in baseline_results:
        r = baseline_results[name]
        baseline_dl_summary.append({
            'Model': name, 'Train R2': r['train_r2'],
            'Test R2': r['test_r2'], 'RMSE': r['rmse'], 'MAE': r['mae']})
summary_df = pd.DataFrame(baseline_dl_summary).sort_values('Test R2',
                                                              ascending=False)
print("\n" + summary_df.to_string(index=False))

# ==============================================================================
# CELLS 10: OPTIMIZED MODELS
# ==============================================================================
print("\n" + "="*70)
print("OPTIMIZED MODELS")
print("="*70)
print(f"Optimizer config: n_agents={opt_n_agents}, max_iter={opt_max_iter}, "
      f"patience={opt_patience}, CV={default_cv_folds}-fold")

optimized_results = {}


def run_ml_optimization(model_name, optimizer_class, optimizer_name):
    """Run ML hyperparameter optimization with reduced parameters."""
    print(f"\nOPTIMIZING: {model_name} with {optimizer_name}")
    print("="*70)
    start_time = time.time()
    config = ML_MODEL_CONFIGS[model_name]
    if config['model'] is None:
        print(f"{model_name} not available, skipping...")
        return None
    bounds = config['bounds']
    dim = len(bounds)
    objective = create_ml_objective(
        model_name, X_train_scaled, y_train_scaled,
        cv_folds=default_cv_folds)
    # OPTIMIZATION: Aggressively reduced optimizer parameters
    if optimizer_name == 'AQPSO':
        optimizer = optimizer_class(
            n_particles=opt_n_agents, max_iter=opt_max_iter, dim=dim,
            bounds=bounds, early_stop_patience=opt_patience)
    elif optimizer_name == 'GWO':
        optimizer = optimizer_class(
            n_wolves=opt_n_agents, max_iter=opt_max_iter, dim=dim,
            bounds=bounds, early_stop_patience=opt_patience)
    else:
        optimizer = optimizer_class(
            n_agents=opt_n_agents, max_iter=opt_max_iter, dim=dim,
            bounds=bounds, early_stop_patience=opt_patience)
    best_params, best_fitness = optimizer.optimize(objective, verbose=True)
    param_dict = config.get('fixed', {}).copy()
    int_params = config.get('int_params', [])
    for i, param in enumerate(config['params']):
        if i in int_params:
            param_dict[param] = int(round(best_params[i]))
        else:
            param_dict[param] = best_params[i]
    print(f"\nOptimization completed ({time.time()-start_time:.1f}s)")
    print(f"Best CV R2: {-best_fitness:.4f}")
    model = config['model'](**param_dict)
    model.fit(X_train_scaled, y_train_scaled)
    y_pred_train = scaler_y.inverse_transform(
        model.predict(X_train_scaled).reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(
        model.predict(X_test_scaled).reshape(-1, 1)).ravel()
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Final: Train R2={train_r2:.4f}, Test R2={test_r2:.4f}, "
          f"RMSE={rmse:.4f}")
    save_checkpoint()
    return {
        'model_name': model_name, 'model_type': 'Optimized',
        'optimizer': optimizer_name, 'model': model,
        'best_params': param_dict,
        'train_r2': train_r2, 'test_r2': test_r2,
        'rmse': rmse, 'mae': mae, 'y_pred_test': y_pred_test}


def run_dl_optimization(model_name, model_class, optimizer_class,
                        optimizer_name):
    """Run DL hyperparameter optimization with reduced parameters."""
    print(f"\nOPTIMIZING: {model_name} with {optimizer_name}")
    print("="*70)
    start_time = time.time()
    config = DL_MODEL_CONFIGS[model_name]
    bounds = config['bounds']
    dim = len(bounds)
    objective = create_dl_objective(
        model_name, X_train_scaled, y_train_scaled, epochs=10)
    # OPTIMIZATION: Aggressively reduced optimizer parameters
    if optimizer_name == 'AQPSO':
        optimizer = optimizer_class(
            n_particles=opt_n_agents, max_iter=opt_max_iter, dim=dim,
            bounds=bounds, early_stop_patience=opt_patience)
    elif optimizer_name == 'GWO':
        optimizer = optimizer_class(
            n_wolves=opt_n_agents, max_iter=opt_max_iter, dim=dim,
            bounds=bounds, early_stop_patience=opt_patience)
    else:
        optimizer = optimizer_class(
            n_agents=opt_n_agents, max_iter=opt_max_iter, dim=dim,
            bounds=bounds, early_stop_patience=opt_patience)
    best_params, best_fitness = optimizer.optimize(objective, verbose=True)
    param_dict = {}
    int_params = config.get('int_params', [])
    for i, param in enumerate(config['params']):
        if i in int_params:
            param_dict[param] = int(round(best_params[i]))
        else:
            param_dict[param] = best_params[i]
    print(f"\nOptimization completed ({time.time()-start_time:.1f}s)")
    print(f"Best val loss: {best_fitness:.6f}")

    # Build final model with optimized hyperparameters
    if model_name == 'SimpleMLP':
        model = model_class(
            input_dim=input_dim,
            hidden_units=[param_dict['hidden_units_1'],
                          param_dict['hidden_units_2']],
            dropout_rate=param_dict['dropout_rate'])
    elif model_name == 'OptimizedMLP':
        model = model_class(
            input_dim=input_dim,
            hidden_layers=[param_dict['hidden_units_1'],
                           param_dict['hidden_units_2'],
                           param_dict['hidden_units_3']],
            dropout_rate=param_dict['dropout_rate'],
            l2_reg=param_dict['l2_reg'])
    else:
        model = model_class(
            input_dim=input_dim,
            hidden_layers=[param_dict['hidden_units_1'],
                           param_dict['hidden_units_2'],
                           param_dict['hidden_units_3']],
            dropout_rate=param_dict['dropout_rate'])

    train_ds = WellLogDataset(X_train_scaled, y_train_scaled)
    val_ds = WellLogDataset(X_test_scaled, y_test_scaled)
    # OPTIMIZATION: batch_size=256
    train_loader = DataLoader(
        train_ds, batch_size=dl_batch_size, shuffle=False,
        drop_last=True, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=dl_batch_size, shuffle=False, num_workers=0)
    # OPTIMIZATION: Train final model with reduced epochs (20 instead of 50)
    model, _ = train_pytorch_model(
        model, train_loader, val_loader, epochs=dl_opt_epochs,
        lr=param_dict['learning_rate'], patience=10, device=device)
    # OPTIMIZATION: Batch prediction
    train_pred = predict_in_batches(model, X_train_scaled, device)
    test_pred = predict_in_batches(model, X_test_scaled, device)
    y_pred_train = scaler_y.inverse_transform(
        train_pred.reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(
        test_pred.reshape(-1, 1)).ravel()
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"\nFinal: Train R2={train_r2:.4f}, Test R2={test_r2:.4f}")
    save_checkpoint()
    return {
        'model_name': model_name, 'model_type': 'Optimized',
        'optimizer': optimizer_name, 'model': model,
        'best_params': param_dict,
        'train_r2': train_r2, 'test_r2': test_r2,
        'rmse': rmse, 'mae': mae, 'y_pred_test': y_pred_test}


# Run optimizations
print("\n--- Random Forest Optimizations ---")
optimized_results['RF_AQPSO'] = run_ml_optimization(
    'RandomForest', AQPSO, 'AQPSO')
optimized_results['RF_GWO'] = run_ml_optimization(
    'RandomForest', GWO, 'GWO')
optimized_results['RF_Hybrid'] = run_ml_optimization(
    'RandomForest', HybridAQPSOGWO, 'Hybrid')

print("\n--- XGBoost Optimizations ---")
if XGBOOST_AVAILABLE:
    optimized_results['XGB_AQPSO'] = run_ml_optimization(
        'XGBoost', AQPSO, 'AQPSO')
    optimized_results['XGB_GWO'] = run_ml_optimization(
        'XGBoost', GWO, 'GWO')
    optimized_results['XGB_Hybrid'] = run_ml_optimization(
        'XGBoost', HybridAQPSOGWO, 'Hybrid')

print("\n--- Simple MLP Optimizations ---")
optimized_results['MLP_AQPSO'] = run_dl_optimization(
    'SimpleMLP', SimpleMLP, AQPSO, 'AQPSO')
optimized_results['MLP_GWO'] = run_dl_optimization(
    'SimpleMLP', SimpleMLP, GWO, 'GWO')
optimized_results['MLP_Hybrid'] = run_dl_optimization(
    'SimpleMLP', SimpleMLP, HybridAQPSOGWO, 'Hybrid')

print("\n--- Optimized MLP Optimizations ---")
optimized_results['OptMLP_AQPSO'] = run_dl_optimization(
    'OptimizedMLP', OptimizedMLP, AQPSO, 'AQPSO')
optimized_results['OptMLP_GWO'] = run_dl_optimization(
    'OptimizedMLP', OptimizedMLP, GWO, 'GWO')
optimized_results['OptMLP_Hybrid'] = run_dl_optimization(
    'OptimizedMLP', OptimizedMLP, HybridAQPSOGWO, 'Hybrid')


# ==============================================================================
# CELLS 11: COMPARISON, VISUALIZATIONS, ENSEMBLES, SAVE
# ==============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON AND ANALYSIS")
print("="*70)

comparison_data = []
for name, result in baseline_results.items():
    if name in ['Random Forest', 'XGBoost', 'LightGBM',
                'Gradient Boosting', 'Extra Trees', 'AdaBoost']:
        comparison_data.append({
            'Model': name, 'Type': 'Baseline', 'Optimizer': '-',
            'Train R2': result['train_r2'], 'Test R2': result['test_r2'],
            'RMSE': result['rmse'], 'MAE': result['mae']})
for name in ['Simple MLP', 'Optimized MLP', 'Residual DNN', 'PINN']:
    if name in baseline_results:
        r = baseline_results[name]
        comparison_data.append({
            'Model': name, 'Type': 'Baseline', 'Optimizer': '-',
            'Train R2': r['train_r2'], 'Test R2': r['test_r2'],
            'RMSE': r['rmse'], 'MAE': r['mae']})
for key, result in optimized_results.items():
    if result:
        comparison_data.append({
            'Model': result['model_name'], 'Type': 'Optimized',
            'Optimizer': result['optimizer'],
            'Train R2': result['train_r2'], 'Test R2': result['test_r2'],
            'RMSE': result['rmse'], 'MAE': result['mae']})

comparison_df = pd.DataFrame(comparison_data).sort_values('Test R2',
                                                           ascending=False)
print("\n" + "="*70)
print("FULL MODEL COMPARISON (Ranked by Test R2)")
print("="*70)
print(comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(run_dir, 'model_comparison.csv'),
                      index=False)
print(f"\nComparison saved to: {run_dir}/model_comparison.csv")

# Overfitting check
print("\n" + "="*70)
print("OVERFITTING ANALYSIS")
print("="*70)
comparison_df['Overfit_Gap'] = (comparison_df['Train R2'] -
                                 comparison_df['Test R2'])
overfit_summary = comparison_df[['Model', 'Type', 'Train R2', 'Test R2',
                                  'Overfit_Gap']].sort_values('Overfit_Gap',
                                                               ascending=False)
print(overfit_summary.to_string(index=False))

# Visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, ax = plt.subplots(figsize=(16, 10))
baseline_df = comparison_df[comparison_df['Type'] == 'Baseline'].copy()
optimized_df = comparison_df[comparison_df['Type'] == 'Optimized'].copy()
models = baseline_df['Model'].unique()
x = np.arange(len(models))
width = 0.35
baseline_r2 = [baseline_df[baseline_df['Model'] == m]['Test R2'].values[0]
               if len(baseline_df[baseline_df['Model'] == m]) > 0 else 0
               for m in models]
bars1 = ax.bar(x - width/2, baseline_r2, width, label='Baseline',
               alpha=0.8, color='steelblue')
best_opt_r2 = [optimized_df[optimized_df['Model'] == m]['Test R2'].max()
               if len(optimized_df[optimized_df['Model'] == m]) > 0 else 0
               for m in models]
bars2 = ax.bar(x + width/2, best_opt_r2, width, label='Best Optimized',
               alpha=0.8, color='coral')
ax.set_xlabel('Models', fontsize=13, fontweight='bold')
ax.set_ylabel('Test R2 Score', fontsize=13, fontweight='bold')
ax.set_title('Baseline vs Optimized Models (R2 Score)', fontsize=14,
             fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'r2_comparison.png'), dpi=300,
            bbox_inches='tight')
print("Saved: r2_comparison.png")
plt.close()

fig, ax = plt.subplots(figsize=(12, 8))
comparison_df['Overfit'] = (comparison_df['Train R2'] -
                             comparison_df['Test R2'])
scatter = ax.scatter(comparison_df['Test R2'], comparison_df['Overfit'],
                     c=pd.Categorical(comparison_df['Type']).codes,
                     s=100, alpha=0.7, cmap='coolwarm', edgecolors='black')
ax.axhline(y=0.10, color='r', linestyle='--', alpha=0.5,
           label='10% Overfit Warning')
ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5,
           label='5% Overfit Threshold')
ax.set_xlabel('Test R2', fontsize=12, fontweight='bold')
ax.set_ylabel('Overfit Gap (Train R2 - Test R2)', fontsize=12,
              fontweight='bold')
ax.set_title('Overfitting Analysis: All Models', fontsize=14,
             fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'overfitting_analysis.png'), dpi=300,
            bbox_inches='tight')
print("Saved: overfitting_analysis.png")
plt.close()

# Ensemble
print("\n" + "="*70)
print("ENSEMBLE METHODS")
print("="*70)
all_predictions = {}
for name, result in baseline_results.items():
    if 'y_pred_test' in result:
        all_predictions[name] = result['y_pred_test']
for key, result in optimized_results.items():
    if result and 'y_pred_test' in result:
        all_predictions[key] = result['y_pred_test']
print(f"\nTotal models for ensemble: {len(all_predictions)}")

if len(all_predictions) > 0:
    pred_matrix = np.array(list(all_predictions.values()))
    # Simple Average Ensemble
    ensemble_simple = np.mean(pred_matrix, axis=0)
    r2_simple = r2_score(y_test, ensemble_simple)
    rmse_simple = np.sqrt(mean_squared_error(y_test, ensemble_simple))
    mae_simple = mean_absolute_error(y_test, ensemble_simple)
    print(f"Simple Average Ensemble: R2={r2_simple:.4f}, "
          f"RMSE={rmse_simple:.4f}, MAE={mae_simple:.4f}")
    # Weighted Average Ensemble
    r2_scores = np.array([r2_score(y_test, pred) for pred in pred_matrix])
    min_r2 = r2_scores.min()
    adjusted_r2 = r2_scores - min_r2 + 0.1
    weights = adjusted_r2 / adjusted_r2.sum()
    ensemble_weighted = np.average(pred_matrix, axis=0, weights=weights)
    r2_weighted = r2_score(y_test, ensemble_weighted)
    rmse_weighted = np.sqrt(mean_squared_error(y_test, ensemble_weighted))
    mae_weighted = mean_absolute_error(y_test, ensemble_weighted)
    print(f"Weighted Average Ensemble: R2={r2_weighted:.4f}, "
          f"RMSE={rmse_weighted:.4f}, MAE={mae_weighted:.4f}")
    # Top-5 Models Ensemble
    top5_idx = np.argsort(r2_scores)[-5:]
    ensemble_top5 = np.mean(pred_matrix[top5_idx], axis=0)
    r2_top5 = r2_score(y_test, ensemble_top5)
    rmse_top5 = np.sqrt(mean_squared_error(y_test, ensemble_top5))
    mae_top5 = mean_absolute_error(y_test, ensemble_top5)
    print(f"Top-5 Models Ensemble: R2={r2_top5:.4f}, "
          f"RMSE={rmse_top5:.4f}, MAE={mae_top5:.4f}")
    # Save ensemble results
    pd.DataFrame({
        'Method': ['Simple Average', 'Weighted Average', 'Top-5 Average'],
        'R2': [r2_simple, r2_weighted, r2_top5],
        'RMSE': [rmse_simple, rmse_weighted, rmse_top5],
        'MAE': [mae_simple, mae_weighted, mae_top5]
    }).to_csv(os.path.join(run_dir, 'ensemble_results.csv'), index=False)

# Save models
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

ml_models_to_save = [
    (f'{name}_baseline', result)
    for name, result in baseline_results.items()
    if name in ['Random Forest', 'XGBoost', 'LightGBM',
                'Gradient Boosting', 'Extra Trees', 'AdaBoost']
]
for key, result in optimized_results.items():
    if result and key.startswith(('RF_', 'XGB_', 'LGBM_', 'GB_',
                                   'ET_', 'ADA_')):
        ml_models_to_save.append((key, result))

saved_count = 0
for name, result in ml_models_to_save:
    if result and 'model' in result:
        try:
            model_path = os.path.join(models_dir, f"{name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            saved_count += 1
        except Exception as e:
            print(f"Error saving {name}: {e}")
print(f"Saved {saved_count} ML models")

saved_dl = 0
for name in ['Simple MLP', 'Optimized MLP', 'Residual DNN', 'PINN']:
    if name in baseline_results:
        try:
            model_path = os.path.join(
                models_dir, f"{name.replace(' ', '_')}_baseline.pt")
            torch.save(baseline_results[name]['model'].state_dict(),
                       model_path)
            saved_dl += 1
        except Exception as e:
            print(f"Error saving {name}: {e}")
for key, result in optimized_results.items():
    if result and key.startswith(('MLP_', 'OptMLP_', 'ResDNN_', 'PINN_')):
        try:
            model_path = os.path.join(models_dir, f"{key}.pt")
            torch.save(result['model'].state_dict(), model_path)
            saved_dl += 1
        except Exception as e:
            print(f"Error saving {key}: {e}")
print(f"Saved {saved_dl} DL models")

# Save comparison DataFrame and config
comparison_df.to_pickle(os.path.join(run_dir, 'comparison_df.pkl'))
print("Saved comparison DataFrame")

config = {
    'random_seed': RANDOM_SEED,
    'target_column': TARGET_COLUMN,
    'input_dim': input_dim,
    'train_samples': X_train_scaled.shape[0],
    'test_samples': X_test_scaled.shape[0],
    'feature_columns': feature_cols,
    'split_type': 'random_80_20',
    'device': str(device),
    'well_column': WELL_COL,
    'depth_column': DEPTH_COL,
    'optimized_params': {
        'n_agents': opt_n_agents,
        'max_iter': opt_max_iter,
        'patience': opt_patience,
        'cv_folds': default_cv_folds,
        'dl_batch_size': dl_batch_size,
        'dl_opt_epochs': dl_opt_epochs,
        'dl_base_epochs': dl_base_epochs
    }
}
with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print("Saved configuration")

# ==============================================================================
# CELL 12: SHAP ANALYSIS
# ==============================================================================
print("\n" + "="*70)
print("SHAP ANALYSIS - Model Interpretability")
print("="*70)

try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP version:", shap.__version__)
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install: pip install shap")

if SHAP_AVAILABLE:
    # Select top ML baseline models for SHAP (tree-based only)
    shap_models = {}
    for name in ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting',
                 'Extra Trees']:
        if name in baseline_results and baseline_results[name] is not None:
            shap_models[name] = baseline_results[name]['model']

    if not shap_models:
        print("No suitable models found for SHAP analysis")
    else:
        # Limit background data for SHAP (memory safety)
        max_bg = min(500, len(X_train_scaled))
        bg_indices = np.random.RandomState(RANDOM_SEED).choice(
            len(X_train_scaled), max_bg, replace=False)
        X_background = X_train_scaled[bg_indices]

        for model_name, model in shap_models.items():
            print(f"\n--- SHAP: {model_name} ---")
            try:
                # TreeExplainer for tree-based models
                if hasattr(model, 'estimators_') or \
                   hasattr(model, 'get_booster') or \
                   hasattr(model, 'trees_'):
                    explainer = shap.TreeExplainer(
                        model, feature_names=feature_cols)
                    # Use subset of test data for SHAP values
                    max_samples = min(500, len(X_test_scaled))
                    sample_idx = np.random.RandomState(
                        RANDOM_SEED).choice(
                        len(X_test_scaled), max_samples, replace=False)
                    X_shap = X_test_scaled[sample_idx]
                    shap_values = explainer.shap_values(X_shap)

                    # Handle multi-output (take first output for regression)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]

                    # 1. SHAP Summary Bar Plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    shap.summary_plot(
                        shap_values, X_shap, feature_names=feature_cols,
                        plot_type='bar', show=False)
                    plt.title(f'SHAP Feature Importance - {model_name}',
                              fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(plots_dir,
                                     f'shap_bar_{model_name.replace(" ", "_")}.png'),
                        dpi=300, bbox_inches='tight')
                    print(f"  Saved: shap_bar_{model_name.replace(' ', '_')}.png")
                    plt.close()

                    # 2. SHAP Summary Beeswarm Plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    shap.summary_plot(
                        shap_values, X_shap, feature_names=feature_cols,
                        show=False)
                    plt.title(
                        f'SHAP Feature Impact Distribution - {model_name}',
                        fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(plots_dir,
                                     f'shap_beeswarm_{model_name.replace(" ", "_")}.png'),
                        dpi=300, bbox_inches='tight')
                    print(f"  Saved: shap_beeswarm_{model_name.replace(' ', '_')}.png")
                    plt.close()

                    # 3. SHAP Heatmap (sample-wise contributions)
                    if len(shap_values) >= 50:
                        fig, ax = plt.subplots(figsize=(14, 10))
                        shap.plots.heatmap(
                            shap.Explanation(
                                shap_values[:50],
                                data=X_shap[:50],
                                feature_names=feature_cols),
                            show=False)
                        plt.title(
                            f'SHAP Heatmap - {model_name} (50 samples)',
                            fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(plots_dir,
                                         f'shap_heatmap_{model_name.replace(" ", "_")}.png'),
                            dpi=300, bbox_inches='tight')
                        print(f"  Saved: shap_heatmap_{model_name.replace(' ', '_')}.png")
                        plt.close()

                    # 4. SHAP Dependence plots for top 3 features
                    mean_shap = np.abs(shap_values).mean(axis=0)
                    top3_idx = np.argsort(mean_shap)[-3:][::-1]
                    for rank, feat_idx in enumerate(top3_idx, 1):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.dependence_plot(
                            feat_idx, shap_values, X_shap,
                            feature_names=feature_cols, show=False)
                        plt.title(
                            f'SHAP Dependence #{rank} - {feature_cols[feat_idx]}',
                            fontsize=12, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(plots_dir,
                                         f'shap_depend_{rank}_{model_name.replace(" ", "_")}_'
                                         f'{feature_cols[feat_idx]}.png'),
                            dpi=300, bbox_inches='tight')
                        plt.close()
                    print(f"  Saved: 3 SHAP dependence plots")

                    # 5. SHAP Waterfall for median-impact sample
                    median_idx = np.argsort(
                        np.abs(shap_values).sum(axis=1))[
                        len(shap_values)//2]
                    fig, ax = plt.subplots(figsize=(12, 8))
                    shap.plots.waterfall(
                        shap.Explanation(
                            shap_values[median_idx],
                            base_values=explainer.expected_value
                            if hasattr(explainer, 'expected_value')
                            else shap_values.mean(),
                            data=X_shap[median_idx],
                            feature_names=feature_cols),
                        show=False)
                    plt.title(
                        f'SHAP Waterfall - {model_name} (Median Sample)',
                        fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(plots_dir,
                                     f'shap_waterfall_{model_name.replace(" ", "_")}.png'),
                        dpi=300, bbox_inches='tight')
                    print(f"  Saved: shap_waterfall_{model_name.replace(' ', '_')}.png")
                    plt.close()

                gc.collect()
            except Exception as e:
                print(f"  SHAP analysis failed for {model_name}: {e}")
                continue

        print("\nSHAP analysis complete!")
else:
    print("SHAP analysis skipped (not installed)")

# ==============================================================================
# CELL 13: LIME + HEATMAP ANALYSIS
# ==============================================================================
print("\n" + "="*70)
print("LIME ANALYSIS + PUBLICATION HEATMAPS")
print("="*70)

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
    print("LIME available")
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install: pip install lime")

if LIME_AVAILABLE:
    # Select top 3 ML models for LIME
    lime_model_list = []
    for name in ['Random Forest', 'XGBoost', 'LightGBM']:
        if name in baseline_results and baseline_results[name] is not None:
            lime_model_list.append(
                (name, baseline_results[name]['model']))

    if lime_model_list and len(X_test_scaled) >= 3:
        # Select 3 representative test samples (min, median, max predicted)
        # Use Random Forest predictions for sample selection
        rf_name, rf_model = lime_model_list[0]
        preds_all = rf_model.predict(X_test_scaled)
        sample_indices = [
            np.argmin(preds_all),
            np.argsort(preds_all)[len(preds_all)//2],
            np.argmax(preds_all)]
        sample_labels = ['Min Prediction', 'Median Prediction',
                         'Max Prediction']

        explainer = LimeTabularExplainer(
            X_train_scaled,
            feature_names=feature_cols,
            mode='regression',
            random_state=RANDOM_SEED)

        for model_name, model in lime_model_list:
            print(f"\n--- LIME: {model_name} ---")
            try:
                for idx, (s_idx, label) in enumerate(
                    zip(sample_indices, sample_labels), 1):
                    exp = explainer.explain_instance(
                        X_test_scaled[s_idx],
                        model.predict,
                        num_features=min(10, len(feature_cols)))

                    fig = exp.as_pyplot_figure()
                    plt.title(
                        f'LIME: {model_name} - {label}\n'
                        f'Sample #{s_idx}, Actual={y_test[s_idx]:.4f}',
                        fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            plots_dir,
                            f'lime_{idx}_{model_name.replace(" ", "_")}_'
                            f'{label.replace(" ", "_")}.png'),
                        dpi=300, bbox_inches='tight')
                    plt.close()
                print(f"  Saved: 3 LIME explanations")
            except Exception as e:
                print(f"  LIME failed for {model_name}: {e}")
                continue
    else:
        print("Not enough models or test data for LIME analysis")
else:
    print("LIME analysis skipped (not installed)")

# ==============================================================================
# HEATMAPS 
# ==============================================================================
print("\n" + "="*70)
print("PUBLICATION-QUALITY HEATMAPS")
print("="*70)

# 1. Feature Correlation Heatmap
print("\n1. Feature Correlation Heatmap")
try:
    fig, ax = plt.subplots(figsize=(max(10, len(feature_cols)*0.5),
                                     max(8, len(feature_cols)*0.4)))
    corr = df_processed[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                annot_kws={'size': 7},
                cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'})
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold',
                 pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'heatmap_feature_correlation.png'),
                dpi=300, bbox_inches='tight')
    print("  Saved: heatmap_feature_correlation.png")
    plt.close()
except Exception as e:
    print(f"  Correlation heatmap failed: {e}")

# 2. Model Performance Heatmap
print("\n2. Model Performance Heatmap")
try:
    perf_pivot = comparison_df.pivot_table(
        index='Model', columns='Type',
        values='Test R2', aggfunc='max')
    if not perf_pivot.empty:
        fig, ax = plt.subplots(figsize=(8, max(6, len(perf_pivot)*0.5)))
        sns.heatmap(perf_pivot, annot=True, fmt='.4f', cmap='YlOrRd',
                    vmin=0, vmax=1, ax=ax,
                    cbar_kws={'shrink': 0.8, 'label': 'Test R2'})
        ax.set_title('Model Performance Heatmap (Test R2)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, 'heatmap_model_performance.png'),
            dpi=300, bbox_inches='tight')
        print("  Saved: heatmap_model_performance.png")
        plt.close()
except Exception as e:
    print(f"  Performance heatmap failed: {e}")

# 3. Prediction Error Heatmap (binned actual vs predicted)
print("\n3. Prediction Error Heatmap")
try:
    for name in ['Random Forest', 'XGBoost']:
        if name in baseline_results and baseline_results[name] is not None:
            y_true = y_test
            y_pred = baseline_results[name]['y_pred_test']
            # Create binned 2D histogram
            n_bins = 50
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Hexbin plot
            hb = axes[0].hexbin(y_true, y_pred, gridsize=30, cmap='YlOrRd',
                                mincnt=1)
            axes[0].plot([y_true.min(), y_true.max()],
                         [y_true.min(), y_true.max()],
                         'b--', lw=2, label='Perfect Prediction')
            axes[0].set_xlabel('Actual PHIE', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Predicted PHIE', fontsize=12,
                               fontweight='bold')
            axes[0].set_title(f'{name}: Actual vs Predicted',
                              fontsize=13, fontweight='bold')
            axes[0].legend()
            plt.colorbar(hb, ax=axes[0], shrink=0.8,
                         label='Sample Count')

            # Residual heatmap
            residuals = y_true - y_pred
            axes[1].hist2d(y_true, residuals, bins=30, cmap='RdYlBu_r')
            axes[1].axhline(y=0, color='k', linestyle='--', lw=2)
            axes[1].set_xlabel('Actual PHIE', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Residual (Actual - Predicted)',
                               fontsize=12, fontweight='bold')
            axes[1].set_title(f'{name}: Residual Distribution',
                              fontsize=13, fontweight='bold')
            plt.colorbar(axes[1].collections[0], ax=axes[1], shrink=0.8,
                         label='Sample Count')

            plt.suptitle(f'Prediction Quality Analysis - {name}',
                         fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir,
                             f'heatmap_prediction_{name.replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
            print(f"  Saved: heatmap_prediction_{name.replace(' ', '_')}.png")
            plt.close()
except Exception as e:
    print(f"  Prediction heatmap failed: {e}")

# 4. SHAP-style Feature Importance Heatmap (across all models)
print("\n4. Cross-Model Feature Importance Heatmap")
try:
    if SHAP_AVAILABLE and shap_models:
        importance_matrix = []
        model_names_imp = []
        for m_name, m_model in shap_models.items():
            try:
                explainer_temp = shap.TreeExplainer(m_model)
                sv = explainer_temp.shap_values(
                    X_test_scaled[:min(300, len(X_test_scaled))])
                if isinstance(sv, list):
                    sv = sv[0]
                importance_matrix.append(
                    np.abs(sv).mean(axis=0))
                model_names_imp.append(m_name)
            except Exception:
                continue

        if importance_matrix:
            imp_df = pd.DataFrame(
                importance_matrix, index=model_names_imp,
                columns=feature_cols)
            # Normalize per model
            imp_df = imp_df.div(imp_df.max(axis=1), axis=0)

            fig, ax = plt.subplots(
                figsize=(max(12, len(feature_cols)*0.6),
                         max(6, len(imp_df)*0.8)))
            sns.heatmap(imp_df, annot=True, fmt='.2f', cmap='YlOrRd',
                        ax=ax, vmin=0, vmax=1,
                        cbar_kws={'shrink': 0.8,
                                  'label': 'Normalized Importance'})
            ax.set_title('Cross-Model Feature Importance Heatmap\n'
                         '(Normalized SHAP values)',
                         fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir,
                             'heatmap_cross_model_importance.png'),
                dpi=300, bbox_inches='tight')
            print("  Saved: heatmap_cross_model_importance.png")
            plt.close()
except Exception as e:
    print(f"  Cross-model importance heatmap failed: {e}")

# 5. Residual Analysis Heatmap (well-wise)
print("\n5. Well-wise Residual Analysis")
try:
    if WELL_COL in df_processed.columns and len(y_test) == len(wells_test):
        rf_pred_test = baseline_results.get(
            'Random Forest', {}).get('y_pred_test', None)
        if rf_pred_test is not None:
            res_df = pd.DataFrame({
                'Well': wells_test,
                'Actual': y_test,
                'Predicted': rf_pred_test,
                'Residual': y_test - rf_pred_test
            })
            well_stats = res_df.groupby('Well').agg({
                'Residual': ['mean', 'std'],
                'Actual': 'count'
            }).round(4)
            well_stats.columns = ['Mean Residual', 'Std Residual', 'N']
            well_stats = well_stats[well_stats['N'] >= 10]

            if len(well_stats) > 0:
                fig, ax = plt.subplots(
                    figsize=(10, max(6, len(well_stats)*0.4)))
                # Color by mean residual value
                colors = well_stats['Mean Residual'].values.reshape(-1, 1)
                sns.heatmap(well_stats[['Mean Residual', 'Std Residual']],
                            annot=True, fmt='.4f', cmap='RdBu_r',
                            center=0, ax=ax,
                            cbar_kws={'shrink': 0.8})
                ax.set_title('Well-wise Residual Analysis (Random Forest)',
                             fontsize=14, fontweight='bold')
                plt.yticks(rotation=0, fontsize=9)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir,
                                 'heatmap_well_residuals.png'),
                    dpi=300, bbox_inches='tight')
                print("  Saved: heatmap_well_residuals.png")
                plt.close()
except Exception as e:
    print(f"  Well residual heatmap failed: {e}")

print("\nAll SHAP, LIME, and Heatmap analyses complete!")
print("="*70)

# ==============================================================================
# BONUS: PER-WELL DEPTH-SPLIT DIAGNOSTIC
# ==============================================================================
print("\n" + "="*70)
print("BONUS: PER-WELL DEPTH-SPLIT DIAGNOSTIC")
print("="*70)
print("(80/20 depth split per well - shows if each well is predictable)")
print("="*70)

per_well_results = []
for well in sorted(df_processed[WELL_COL].unique()):
    well_df = df_processed[df_processed[WELL_COL] == well].copy().sort_values(
        DEPTH_COL)
    if len(well_df) < 50:
        continue
    Xw = well_df[feature_cols].values
    yw = well_df[TARGET_COLUMN].values
    split_idx = int(len(Xw) * 0.8)
    if split_idx < 10 or len(Xw) - split_idx < 5:
        continue
    Xw_train, Xw_test = Xw[:split_idx], Xw[split_idx:]
    yw_train, yw_test = yw[:split_idx], yw[split_idx:]

    sx = RobustScaler()
    Xw_train_s = sx.fit_transform(Xw_train)
    Xw_test_s = sx.transform(Xw_test)
    sy = RobustScaler()
    yw_train_s = sy.fit_transform(yw_train.reshape(-1, 1)).ravel()
    yw_test_s = sy.transform(yw_test.reshape(-1, 1)).ravel()

    m = RandomForestRegressor(n_estimators=100, max_depth=15,
                              min_samples_leaf=3, random_state=RANDOM_SEED,
                              n_jobs=-1)
    m.fit(Xw_train_s, yw_train_s)
    yw_pred = sy.inverse_transform(
        m.predict(Xw_test_s).reshape(-1, 1)).ravel()
    r2 = r2_score(yw_test, yw_pred)
    per_well_results.append({'Well': well, 'Samples': len(well_df),
                              'Test_R2': r2})
    print(f"{well:<30} | n={len(well_df):>6} | Test R = {r2:>7.4f}")

if per_well_results:
    pw_df = pd.DataFrame(per_well_results)
    print(f"\n{'='*70}")
    print(f"Per-Well Diagnostic Summary:")
    print(f"  Mean Test R:  {pw_df['Test_R2'].mean():.4f}")
    print(f"  Median Test R:{pw_df['Test_R2'].median():.4f}")
    print(f"  Wells with R>0.5: {sum(pw_df['Test_R2'] > 0.5)}/{len(pw_df)}")
    print(f"  Wells with R<0:   {sum(pw_df['Test_R2'] < 0)}/{len(pw_df)}")
    pw_df.to_csv(os.path.join(run_dir, 'per_well_diagnostic.csv'),
                  index=False)
    print(f"Saved per-well diagnostic to: {run_dir}/per_well_diagnostic.csv")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print("\nTop 10 Models (by Test R2):")
print("-"*70)
for idx, row in comparison_df.head(10).iterrows():
    print(f"{row['Model']:<20} | {row['Type']:<10} | {row['Optimizer']:<8} | "
          f"R2: {row['Test R2']:.4f} | RMSE: {row['RMSE']:.4f} | "
          f"Overfit: {row['Overfit_Gap']:+.3f}")

print("\n" + "="*70)
print("ENSEMBLE RESULTS:")
print("="*70)
if len(all_predictions) > 0:
    print(f"Simple Average: R2 = {r2_simple:.4f}, RMSE = {rmse_simple:.4f}")
    print(f"Weighted Average: R2 = {r2_weighted:.4f}, "
          f"RMSE = {rmse_weighted:.4f}")
    print(f"Top-5 Average: R2 = {r2_top5:.4f}, RMSE = {rmse_top5:.4f}")

print("\n" + "="*70)
print("OPTIMIZATIONS APPLIED:")
print("="*70)
print(f"  - n_particles/wolves/agents: 30 -> {opt_n_agents} "
      f"(6x fewer evaluations)")
print(f"  - max_iterations: 50 -> {opt_max_iter} (5x fewer iterations)")
print(f"  - CV folds: 5 -> {default_cv_folds} (40% faster CV)")
print(f"  - stagnation patience: 10 -> {opt_patience} (earlier stopping)")
print(f"  - MLP batch_size: 64 -> {dl_batch_size} (better GPU utilization)")
print(f"  - DL opt epochs: 50 -> {dl_opt_epochs} (2.5x fewer)")
print(f"  - DL base epochs: 30 -> {dl_base_epochs} (1.5x fewer)")
print(f"  - Per-well normalization: REMOVED (massive speedup)")
print(f"  - Parallel objective eval: joblib.Parallel (threading)")
print(f"  - XGBoost: tree_method='hist' (fast CPU histogram)")
print(f"  - LightGBM: GPU device mode")
print(f"  - RF/ExtraTrees: cuML GPU when available")
print(f"  - PyTorch: AMP + TF32 + set_to_none (torch.compile removed - needs Triton)")
print(f"  - Checkpointing: auto-save after each model + atexit")
print("="*70)
print(f"All results saved to: {run_dir}")
print("="*70)
print("\nFiles generated:")
print(f" - {run_dir}/ALL_combined_final.csv")
print(f" - {run_dir}/model_comparison.csv")
print(f" - {run_dir}/ensemble_results.csv")
print(f" - {run_dir}/per_well_diagnostic.csv")
print(f" - {run_dir}/config.json")
print(f" - {run_dir}/checkpoint.json")
print(f" - {plots_dir}/*.png")
print(f" - {models_dir}/*.pkl (ML models)")
print(f" - {models_dir}/*.pt (PyTorch DL models)")
print("\n" + "="*70)
print("PIPELINE COMPLETED SUCCESSFULLY! (GPU ACCELERATED)")
print(f"Total experiments: {len(comparison_data)} models evaluated")
print(f"Device used: {device}")
print("="*70)
