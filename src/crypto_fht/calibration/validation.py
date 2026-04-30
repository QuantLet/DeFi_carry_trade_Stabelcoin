"""
Model validation and diagnostics for calibrated Lévy processes.

Provides tools to assess whether the fitted model adequately
describes the observed data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from crypto_fht.core.levy_process import LevyParameters, simulate_paths

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ValidationResult:
    """Results from model validation.

    Attributes:
        ks_statistic: Kolmogorov-Smirnov statistic for return distribution.
        ks_pvalue: KS test p-value.
        ad_statistic: Anderson-Darling statistic.
        ljung_box_statistic: Ljung-Box Q statistic for autocorrelation.
        ljung_box_pvalue: Ljung-Box p-value.
        mean_residual: Mean of standardized residuals.
        std_residual: Std of standardized residuals.
        skewness_residual: Skewness of residuals.
        kurtosis_residual: Excess kurtosis of residuals.
        jump_detection_accuracy: Accuracy of jump detection (if jumps known).
        is_valid: Overall validation assessment.
    """

    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float
    ljung_box_statistic: float
    ljung_box_pvalue: float
    mean_residual: float
    std_residual: float
    skewness_residual: float
    kurtosis_residual: float
    jump_detection_accuracy: float | None
    is_valid: bool

    def summary(self) -> str:
        """Return formatted summary."""
        return f"""
Model Validation Results
========================
Distribution Tests:
  KS statistic: {self.ks_statistic:.4f} (p={self.ks_pvalue:.4f})
  AD statistic: {self.ad_statistic:.4f}

Autocorrelation:
  Ljung-Box Q: {self.ljung_box_statistic:.4f} (p={self.ljung_box_pvalue:.4f})

Residual Statistics:
  Mean: {self.mean_residual:.4f}
  Std:  {self.std_residual:.4f}
  Skew: {self.skewness_residual:.4f}
  Kurt: {self.kurtosis_residual:.4f}

Overall Assessment: {'VALID' if self.is_valid else 'INVALID'}
"""


class ModelValidator:
    """Validator for calibrated Lévy process models.

    Performs various statistical tests to assess model fit.

    Attributes:
        params: Calibrated parameters.
        dt: Time step.

    Example:
        >>> validator = ModelValidator(calibrated_params, dt=1/365)
        >>> result = validator.validate(observed_returns)
        >>> print(result.summary())
    """

    def __init__(self, params: LevyParameters, dt: float = 1.0) -> None:
        """Initialize validator.

        Args:
            params: Calibrated Lévy parameters.
            dt: Time step.
        """
        self.params = params
        self.dt = dt

    def validate(
        self,
        returns: NDArray[np.floating],
        significance_level: float = 0.05,
    ) -> ValidationResult:
        """Run full validation suite.

        Args:
            returns: Observed return data.
            significance_level: Significance level for tests.

        Returns:
            ValidationResult with all test results.
        """
        # Compute standardized residuals
        residuals = self._compute_residuals(returns)

        # Distribution tests
        ks_stat, ks_pval = self._ks_test(residuals)
        ad_stat = self._anderson_darling(residuals)

        # Autocorrelation test
        lb_stat, lb_pval = self._ljung_box_test(residuals)

        # Residual statistics
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))
        skew_res = float(stats.skew(residuals))
        kurt_res = float(stats.kurtosis(residuals))

        # Overall validity assessment
        is_valid = (
            ks_pval > significance_level
            and lb_pval > significance_level
            and abs(mean_res) < 0.1
            and abs(std_res - 1) < 0.2
        )

        return ValidationResult(
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ad_statistic=ad_stat,
            ljung_box_statistic=lb_stat,
            ljung_box_pvalue=lb_pval,
            mean_residual=mean_res,
            std_residual=std_res,
            skewness_residual=skew_res,
            kurtosis_residual=kurt_res,
            jump_detection_accuracy=None,
            is_valid=is_valid,
        )

    def _compute_residuals(
        self, returns: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute standardized residuals.

        Under the model: X = μΔt + σΔW - J
        Residual: (X - μΔt + E[J]) / (σ√Δt)
        """
        expected_jump = self.params.lambda_ * self.dt * self.params.mean_jump_size
        mean = self.params.mu * self.dt - expected_jump
        std = self.params.sigma * np.sqrt(self.dt)

        if std < 1e-10:
            return returns - mean

        return (returns - mean) / std

    def _ks_test(
        self, residuals: NDArray[np.floating]
    ) -> tuple[float, float]:
        """Kolmogorov-Smirnov test against standard normal.

        Tests if residuals follow N(0,1) distribution.
        """
        stat, pval = stats.kstest(residuals, "norm")
        return float(stat), float(pval)

    def _anderson_darling(self, residuals: NDArray[np.floating]) -> float:
        """Anderson-Darling test statistic.

        More sensitive to tails than KS test.
        """
        result = stats.anderson(residuals, dist="norm")
        return float(result.statistic)

    def _ljung_box_test(
        self,
        residuals: NDArray[np.floating],
        lags: int = 10,
    ) -> tuple[float, float]:
        """Ljung-Box test for autocorrelation.

        Tests if residual autocorrelations are jointly zero.
        """
        n = len(residuals)
        if n < lags + 1:
            return 0.0, 1.0

        # Compute autocorrelations
        acf = np.correlate(residuals - np.mean(residuals), residuals - np.mean(residuals), mode="full")
        acf = acf[n - 1 :] / acf[n - 1]  # Normalize

        # Ljung-Box Q statistic
        Q = n * (n + 2) * np.sum(acf[1 : lags + 1] ** 2 / (n - np.arange(1, lags + 1)))

        # p-value from chi-squared distribution
        pval = 1 - stats.chi2.cdf(Q, lags)

        return float(Q), float(pval)

    def simulate_and_compare(
        self,
        returns: NDArray[np.floating],
        n_simulations: int = 1000,
    ) -> dict[str, tuple[float, float]]:
        """Compare observed statistics to simulated distribution.

        Args:
            returns: Observed returns.
            n_simulations: Number of simulated paths.

        Returns:
            Dictionary mapping statistic name to (observed, simulated_mean).
        """
        n = len(returns)

        # Observed statistics
        obs_mean = float(np.mean(returns))
        obs_std = float(np.std(returns))
        obs_skew = float(stats.skew(returns))
        obs_kurt = float(stats.kurtosis(returns))
        obs_min = float(np.min(returns))

        # Simulate
        _, sim_paths = simulate_paths(
            self.params,
            x0=0.0,
            t_max=n * self.dt,
            dt=self.dt,
            n_paths=n_simulations,
        )

        # Compute returns from paths
        sim_returns = np.diff(sim_paths, axis=1)

        # Simulated statistics
        sim_means = np.mean(sim_returns, axis=1)
        sim_stds = np.std(sim_returns, axis=1)
        sim_skews = stats.skew(sim_returns, axis=1)
        sim_kurts = stats.kurtosis(sim_returns, axis=1)
        sim_mins = np.min(sim_returns, axis=1)

        return {
            "mean": (obs_mean, float(np.mean(sim_means))),
            "std": (obs_std, float(np.mean(sim_stds))),
            "skewness": (obs_skew, float(np.mean(sim_skews))),
            "kurtosis": (obs_kurt, float(np.mean(sim_kurts))),
            "min": (obs_min, float(np.mean(sim_mins))),
        }


def cross_validate_calibration(
    returns: NDArray[np.floating],
    dt: float,
    n_folds: int = 5,
) -> list[ValidationResult]:
    """Cross-validate calibration across time folds.

    Splits data chronologically and validates out-of-sample.

    Args:
        returns: Full return series.
        dt: Time step.
        n_folds: Number of folds.

    Returns:
        List of ValidationResults for each fold.
    """
    from crypto_fht.calibration.mle_estimator import LevyMLEEstimator

    n = len(returns)
    fold_size = n // n_folds
    results = []

    for i in range(n_folds - 1):
        # Train on folds 0..i, test on fold i+1
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min((i + 2) * fold_size, n)

        train_data = returns[:train_end]
        test_data = returns[test_start:test_end]

        # Calibrate on training data
        estimator = LevyMLEEstimator(dt=dt)
        cal_result = estimator.fit(train_data)

        # Validate on test data
        validator = ModelValidator(cal_result.params, dt=dt)
        val_result = validator.validate(test_data)

        results.append(val_result)

    return results
