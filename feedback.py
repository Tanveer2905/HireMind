"""
feedback.py — Recruiter Feedback Learning System
Stores recruiter decisions (shortlist/reject) and trains a lightweight model
to adjust ranking scores over time based on learned preferences.
"""

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from utils import FEEDBACK_DB_PATH, FEEDBACK_MODEL_PATH

logger = logging.getLogger(__name__)

# Minimum feedback samples before training a model
MIN_TRAINING_SAMPLES = 20

# Maximum score adjustment from the feedback model (prevents runaway bias)
MAX_ADJUSTMENT = 0.10


class FeedbackEngine:
    """
    Learns from recruiter decisions to improve ranking over time.

    Stores shortlist/reject actions and trains a lightweight classifier
    on the scoring features to predict recruiter preferences.
    """

    def __init__(self):
        self.feedback_db: list[dict] = []
        self.model = None
        self._model_loaded = False
        self._load_feedback()
        self._load_model()

    # ------------------------------------------------------------------
    # Feedback storage
    # ------------------------------------------------------------------
    def _load_feedback(self) -> None:
        """Load feedback database from disk."""
        if FEEDBACK_DB_PATH.exists():
            try:
                with open(FEEDBACK_DB_PATH, "r", encoding="utf-8") as f:
                    self.feedback_db = json.load(f)
                logger.info(f"Loaded {len(self.feedback_db)} feedback records")
            except (json.JSONDecodeError, IOError):
                self.feedback_db = []

    def _save_feedback(self) -> None:
        """Save feedback database to disk."""
        FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(FEEDBACK_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(self.feedback_db, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to save feedback: {e}")

    def record_feedback(
        self,
        filename: str,
        action: str,
        scores: dict[str, float],
        jd_text: str = "",
    ) -> dict[str, Any]:
        """
        Record a recruiter decision.

        Args:
            filename: Resume filename
            action: "shortlisted" or "rejected"
            scores: Dict of score components {semantic_score, skill_score, ...}
            jd_text: Job description text (hashed for grouping)

        Returns:
            Status dict with feedback count and training eligibility
        """
        if action not in ("shortlisted", "rejected"):
            return {"error": f"Invalid action: {action}. Use 'shortlisted' or 'rejected'."}

        # Generate JD hash for grouping feedback by job
        jd_hash = hashlib.md5(jd_text.encode("utf-8")).hexdigest()[:8] if jd_text else "unknown"

        record = {
            "filename": filename,
            "action": action,
            "jd_hash": jd_hash,
            "scores": {
                "semantic_score": scores.get("semantic_score", 0),
                "skill_score": scores.get("skill_score", 0),
                "experience_score": scores.get("experience_score", 0),
                "recency_score": scores.get("recency_score", 0),
                "keyword_score": scores.get("keyword_score", 0),
                "final_score": scores.get("final_score", 0),
                "skill_match_pct": scores.get("skill_match_pct", 0),
            },
            "timestamp": time.time(),
        }

        # Check for duplicate (same file + same JD)
        existing_idx = None
        for i, fb in enumerate(self.feedback_db):
            if fb["filename"] == filename and fb["jd_hash"] == jd_hash:
                existing_idx = i
                break

        if existing_idx is not None:
            self.feedback_db[existing_idx] = record
            logger.info(f"Updated feedback for {filename}: {action}")
        else:
            self.feedback_db.append(record)
            logger.info(f"Recorded feedback for {filename}: {action}")

        self._save_feedback()

        # Check if we should retrain
        can_train = len(self.feedback_db) >= MIN_TRAINING_SAMPLES
        if can_train:
            self.train()

        return {
            "recorded": True,
            "action": action,
            "filename": filename,
            "total_feedback": len(self.feedback_db),
            "can_train": can_train,
            "model_active": self.model is not None,
        }

    def get_feedback_stats(self) -> dict[str, Any]:
        """Return feedback statistics."""
        shortlisted = sum(1 for fb in self.feedback_db if fb["action"] == "shortlisted")
        rejected = sum(1 for fb in self.feedback_db if fb["action"] == "rejected")
        return {
            "total": len(self.feedback_db),
            "shortlisted": shortlisted,
            "rejected": rejected,
            "model_active": self.model is not None,
            "training_threshold": MIN_TRAINING_SAMPLES,
            "ready_to_train": len(self.feedback_db) >= MIN_TRAINING_SAMPLES,
        }

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    def train(self) -> bool:
        """
        Train a lightweight classifier on accumulated feedback.

        Uses LightGBM if available, falls back to LogisticRegression.
        Requires at least MIN_TRAINING_SAMPLES records.

        Returns:
            True if training succeeded
        """
        if len(self.feedback_db) < MIN_TRAINING_SAMPLES:
            logger.info(
                f"Not enough feedback to train ({len(self.feedback_db)}/{MIN_TRAINING_SAMPLES})"
            )
            return False

        # Build training data
        features = []
        labels = []
        for fb in self.feedback_db:
            s = fb["scores"]
            feature_vec = [
                s.get("semantic_score", 0),
                s.get("skill_score", 0),
                s.get("experience_score", 0),
                s.get("recency_score", 0),
                s.get("keyword_score", 0),
                s.get("skill_match_pct", 0) / 100.0,  # normalize to 0-1
            ]
            features.append(feature_vec)
            labels.append(1 if fb["action"] == "shortlisted" else 0)

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # Check class balance
        positive_ratio = y.mean()
        if positive_ratio < 0.1 or positive_ratio > 0.9:
            logger.warning(
                f"Imbalanced feedback (positive ratio: {positive_ratio:.2f}). "
                "Model may not be reliable."
            )

        # Try LightGBM first
        try:
            import lightgbm as lgb

            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 8,
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 50,
                "min_child_samples": 5,
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X, y)
            self.model = model
            logger.info("Feedback model trained with LightGBM")

        except (ImportError, Exception) as e:
            logger.warning(f"LightGBM failed ({e}), falling back to LogisticRegression")
            try:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=200, C=1.0)
                model.fit(X, y)
                self.model = model
                logger.info("Feedback model trained with LogisticRegression")
            except ImportError:
                logger.error("Neither LightGBM nor scikit-learn available for feedback model")
                return False

        # Save model
        self._save_model()
        return True

    def _save_model(self) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            return
        try:
            import joblib
            FEEDBACK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, str(FEEDBACK_MODEL_PATH))
            logger.info(f"Feedback model saved to {FEEDBACK_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to save feedback model: {e}")

    def _load_model(self) -> None:
        """Load a previously trained model from disk."""
        if self._model_loaded:
            return
        self._model_loaded = True

        if not FEEDBACK_MODEL_PATH.exists():
            return

        try:
            import joblib
            self.model = joblib.load(str(FEEDBACK_MODEL_PATH))
            logger.info("Loaded feedback model from disk")
        except Exception as e:
            logger.warning(f"Failed to load feedback model: {e}")
            self.model = None

    # ------------------------------------------------------------------
    # Score adjustment
    # ------------------------------------------------------------------
    def predict_adjustment(self, scores: dict[str, float]) -> float:
        """
        Predict a score adjustment based on learned preferences.

        Args:
            scores: Dict of score components for a candidate

        Returns:
            Float adjustment in [-MAX_ADJUSTMENT, +MAX_ADJUSTMENT]
            Returns 0.0 if no model is trained.
        """
        if self.model is None:
            return 0.0

        feature_vec = np.array([[
            scores.get("semantic_score", 0),
            scores.get("skill_score", 0),
            scores.get("experience_score", 0),
            scores.get("recency_score", 0),
            scores.get("keyword_score", 0),
            scores.get("skill_match_pct", 0) / 100.0,
        ]], dtype=np.float32)

        try:
            # Get probability of being shortlisted
            proba = self.model.predict_proba(feature_vec)[0][1]

            # Map probability to adjustment: 0.5 (neutral) → 0, 1.0 → +MAX, 0.0 → -MAX
            adjustment = (proba - 0.5) * 2 * MAX_ADJUSTMENT
            adjustment = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adjustment))

            return round(adjustment, 4)

        except Exception as e:
            logger.warning(f"Feedback prediction failed: {e}")
            return 0.0

    def reset(self) -> dict[str, Any]:
        """Reset all feedback data and the trained model."""
        self.feedback_db = []
        self.model = None

        # Delete files
        if FEEDBACK_DB_PATH.exists():
            FEEDBACK_DB_PATH.unlink()
        if FEEDBACK_MODEL_PATH.exists():
            FEEDBACK_MODEL_PATH.unlink()

        logger.info("Feedback system reset")
        return {"reset": True, "message": "All feedback data and model cleared."}
