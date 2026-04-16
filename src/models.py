from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    spearmanr = None

RANDOM_STATE = 42


def get_torch_modules() -> tuple[Any, Any, Any, Any]:
    """Import torch lazily to avoid hard failures when it is unavailable or restricted."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:  # pragma: no cover
        return None, None, None, None
    return torch, nn, DataLoader, TensorDataset


def build_sklearn_model(model_name: str, task: str) -> Any:
    """Build a baseline or tree model."""
    if model_name == "ridge":
        if task == "regression":
            estimator = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        else:
            estimator = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if model_name == "tree":
        try:
            import xgboost as xgb
        except ImportError:  # pragma: no cover
            xgb = None

        if xgb is not None:
            estimator = (
                xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                )
                if task == "regression"
                else xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                )
            )
        else:
            estimator = (
                GradientBoostingRegressor(random_state=RANDOM_STATE)
                if task == "regression"
                else GradientBoostingClassifier(random_state=RANDOM_STATE)
            )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])

    raise ValueError(f"Unsupported model_name: {model_name}")


@dataclass
class TorchTrainingResult:
    model: Any
    scaler: StandardScaler
    imputer: SimpleImputer
    checkpoint_path: Path | None


class TorchMLPWrapper:
    """PyTorch MLP with time-based validation and early stopping."""

    def __init__(
        self,
        task: str,
        input_dim: int,
        checkpoint_path: Path,
        hidden_dims: tuple[int, int] = (64, 32),
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 15,
    ) -> None:
        self.task = task
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        torch, _, _, _ = get_torch_modules()
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = None

    def _build_network(self) -> Any:
        _, nn, _, _ = get_torch_modules()
        output_dim = 1
        layers = [
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], output_dim),
        ]
        return nn.Sequential(*layers)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> "TorchMLPWrapper":
        torch, nn, DataLoader, TensorDataset = get_torch_modules()
        if torch is None:
            raise ImportError("PyTorch is not installed.")

        x_train_arr = self.imputer.fit_transform(x_train)
        x_val_arr = self.imputer.transform(x_val)
        x_train_arr = self.scaler.fit_transform(x_train_arr)
        x_val_arr = self.scaler.transform(x_val_arr)

        train_features = torch.tensor(x_train_arr, dtype=torch.float32)
        val_features = torch.tensor(x_val_arr, dtype=torch.float32)

        if self.task == "regression":
            train_labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            val_labels = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
            loss_fn = nn.MSELoss()
        else:
            train_labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            val_labels = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
            loss_fn = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=self.batch_size, shuffle=False)
        self.model = self._build_network().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_metric = np.inf
        best_state = None
        patience_left = self.patience

        for _ in range(self.max_epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(val_features.to(self.device)).cpu().numpy().reshape(-1)
            metric = (
                mean_squared_error(y_val.values, val_preds)
                if self.task == "regression"
                else log_loss(y_val.values, 1.0 / (1.0 + np.exp(-val_preds)), labels=[0, 1])
            )
            if metric < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(self.model.state_dict())
                patience_left = self.patience
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, self.checkpoint_path)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        torch, _, _, _ = get_torch_modules()
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        x_arr = self.imputer.transform(x)
        x_arr = self.scaler.transform(x_arr)
        features = torch.tensor(x_arr, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(features).cpu().numpy().reshape(-1)
        if self.task == "classification":
            preds = 1.0 / (1.0 + np.exp(-preds))
        return preds


def build_mlp_model(task: str, input_dim: int, checkpoint_path: Path) -> Any:
    """Build a PyTorch MLP when available, otherwise a sklearn fallback."""
    torch, _, _, _ = get_torch_modules()
    if torch is not None:
        return TorchMLPWrapper(task=task, input_dim=input_dim, checkpoint_path=checkpoint_path)

    estimator = (
        MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, random_state=RANDOM_STATE, max_iter=300)
        if task == "regression"
        else MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, random_state=RANDOM_STATE, max_iter=300)
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def fit_and_predict(
    model_name: str,
    task: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    checkpoint_path: Path,
) -> tuple[Any, np.ndarray]:
    """Fit a model and return out-of-sample predictions."""
    if model_name in {"ridge", "tree"}:
        model = build_sklearn_model(model_name=model_name, task=task)
        model.fit(x_train, y_train)
        if task == "classification" and hasattr(model, "predict_proba"):
            preds = model.predict_proba(x_test)[:, 1]
        else:
            preds = model.predict(x_test)
        return model, np.asarray(preds)

    if model_name == "mlp":
        model = build_mlp_model(task=task, input_dim=x_train.shape[1], checkpoint_path=checkpoint_path)
        if isinstance(model, TorchMLPWrapper):
            model.fit(x_train, y_train, x_val, y_val)
            preds = model.predict(x_test)
        else:
            model.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
            preds = model.predict_proba(x_test)[:, 1] if task == "classification" else model.predict(x_test)
        return model, np.asarray(preds)

    raise ValueError(f"Unsupported model_name: {model_name}")


def extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    """Extract feature importance when the underlying model exposes it."""
    inner_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
    importances = getattr(inner_model, "feature_importances_", None)
    if importances is None:
        return None
    importance_frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    return importance_frame.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_spearman(y_true: pd.Series, y_pred: pd.Series) -> float:
    if spearmanr is None:
        return float("nan")
    value = spearmanr(y_true, y_pred, nan_policy="omit").statistic
    return float(value) if value is not None else float("nan")
