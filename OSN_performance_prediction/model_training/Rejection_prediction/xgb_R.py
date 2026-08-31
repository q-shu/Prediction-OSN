import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import joblib
import optuna  # Import Optuna for Bayesian Optimization

# --- Global Configuration ---
np.seterr(over='ignore')
plt.rc('font', family='Arial')
plt.rcParams['figure.dpi'] = 150


def prediction(data_path, plt_name, group_col='Paper_ID'):
    """
    XGBoost prediction workflow: Group-aware data partitioning, Bayesian optimization, training, saving, and plotting.

    :param data_path: Path to the dataset (Excel file)
    :param plt_name: File name for saving the output plot
    :param group_col: Column name representing literature/group identity (e.g., 'Paper_ID', 'DOI')
    """

    # 1. Load data and separate features, target, and grouping metadata
    df = pd.read_excel(data_path)

    if group_col in df.columns:
        groups_all = df[group_col].values
        # Exclude the group column and target column (last column) from input features X
        feature_cols = [c for c in df.columns if c not in [group_col, df.columns[-1]]]
        X_all = df[feature_cols]
        Y_all = df.iloc[:, -1]
    else:
        raise ValueError(f"Group column '{group_col}' was not found in the dataset. "
                         f"Please specify the correct column name.")

    # 2. Outer split: Use GroupShuffleSplit to strictly isolate literature sources into train/test sets (8:2)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=40)
    train_idx, test_idx = next(gss.split(X_all, Y_all, groups=groups_all))

    X_train, X_test = X_all.iloc[train_idx].reset_index(drop=True), X_all.iloc[test_idx].reset_index(drop=True)
    Y_train, Y_test = Y_all.iloc[train_idx].reset_index(drop=True), Y_all.iloc[test_idx].reset_index(drop=True)
    groups_train = groups_all[train_idx]
    groups_test = groups_all[test_idx]

    print(f"Data partitioning complete:")
    print(f" - Training set: {len(X_train)} samples across {len(np.unique(groups_train))} studies")
    print(f" - Testing set: {len(X_test)} samples across {len(np.unique(groups_test))} unseen studies")
    print(f" - Study overlap check (must be 0): {len(set(groups_train).intersection(set(groups_test)))}")

    # 3. Define objective function for Bayesian Optimization using GroupKFold
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0)
        }

        # 5-fold Group-based Cross-Validation
        gkf = GroupKFold(n_splits=5)
        r2_scores = []

        for train_fold_idx, val_fold_idx in gkf.split(X_train, Y_train, groups=groups_train):
            X_t, X_v = X_train.iloc[train_fold_idx], X_train.iloc[val_fold_idx]
            Y_t, Y_v = Y_train.iloc[train_fold_idx], Y_train.iloc[val_fold_idx]

            model = xgb.XGBRegressor(
                **param,
                random_state=42,
                objective='reg:squarederror',
                early_stopping_rounds=50
            )
            model.fit(X_t, Y_t, eval_set=[(X_v, Y_v)], verbose=False)
            preds = model.predict(X_v)
            r2_scores.append(r2_score(Y_v, preds))

        return np.mean(r2_scores)

    # 4. Perform Bayesian Optimization
    print("\nStarting Group-aware Bayesian Optimization with Optuna for XGBoost...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    print(f"Best parameters found: {study.best_params}")

    # Retrain the final model on the full training set using optimal parameters
    best_model = xgb.XGBRegressor(
        **study.best_params,
        random_state=42,
        objective='reg:squarederror'
    )
    best_model.fit(X_train, Y_train)

    # 5. Save the trained model
    model_dir = "../../model/rej"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_file = os.path.join(model_dir, "xgb.joblib")
    joblib.dump(best_model, model_file)
    print(f"Model saved to: {model_file}")

    # 6. Model evaluation on training and unseen test groups
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    r2_test = r2_score(Y_test, y_pred_test)
    mae_test = mean_absolute_error(Y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(Y_test, y_pred_test))

    print(f"\n[Final Evaluation on Unseen Literature (Group Split)]")
    print(f"Test R²: {r2_test:.4f}, Test MAE: {mae_test:.4f}, Test RMSE: {rmse_test:.4f}")

    # 7. Visualization
    plot_regression_results(Y_train, y_pred_train, Y_test, y_pred_test, r2_test, mae_test, rmse_test, plt_name)


def plot_regression_results(y_true_train, y_pred_train, y_true_test, y_pred_test, r2, mae, rmse, save_name):
    """
    Plot actual vs predicted regression scatter plot and save as a high-resolution image.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    all_data = np.concatenate([y_true_train, y_pred_train, y_true_test, y_pred_test])
    low_lim, high_lim = np.floor(all_data.min()) - 0.5, np.ceil(all_data.max()) + 0.5
    ax.set_xlim(low_lim, high_lim)
    ax.set_ylim(low_lim, high_lim)
    ax.set_aspect('equal')

    # Reference parity line and grid
    ax.plot([low_lim, high_lim], [low_lim, high_lim], color='grey', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, alpha=0.5, zorder=0)

    # Scatter points for training and testing sets
    ax.scatter(y_true_train, y_pred_train, c='#4E79A7', s=70, alpha=0.7, edgecolors='k', linewidths=0.6,
               label='Training data', zorder=2)
    ax.scatter(y_true_test, y_pred_test, c='#E15759', s=70, alpha=0.8, edgecolors='k', linewidths=0.6,
               label='Testing data (Unseen studies)', zorder=3)

    ax.set_xlabel(r'Actual Rejection', fontsize=15, labelpad=10)
    ax.set_ylabel(r'Predicted Rejection', fontsize=15, labelpad=10)
    ax.text(0.04, 0.94, 'XGBoost', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    # Metrics annotation box
    stats_text = f'$R^2 = {r2:.4f}$\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='lightgrey')
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=14, linespacing=1.5, ha='right', va='bottom',
            bbox=props)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', direction='in', length=6)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.88), frameon=False, fontsize=13)

    # Save figure
    save_dir = "../../pic"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.tight_layout()
    fig.savefig(f"{save_dir}/{save_name}.png", dpi=500)
    plt.show()


if __name__ == '__main__':
    # Make sure 'group_col' matches the column name in your Excel representing literature source/ID
    prediction(
        data_path='../../data/your_data.xlsx',
        plt_name='xgb',
        group_col='Paper_ID'  # Replace with the actual column header representing literature/group IDs
    )