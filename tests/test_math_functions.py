"""Test basic mathematical functions used in the project."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


class TestMathematicalFunctions:
    """Test core mathematical functions."""

    def test_numpy_basic_operations(self):
        """Test basic numpy operations used in the project."""
        # Test gaussian distribution generation
        mean, std = 0.0, 1.0
        samples = np.random.normal(mean, std, 1000)

        assert len(samples) == 1000
        assert abs(np.mean(samples) - mean) < 0.1  # Should be close to expected mean
        assert abs(np.std(samples) - std) < 0.1  # Should be close to expected std

    def test_gaussian_pdf(self):
        """Test Gaussian PDF calculation."""

        def gaussian_pdf(x, mean=0, std=1):
            return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x - mean) / std) ** 2
            )

        # Test at mean (should be maximum)
        pdf_at_mean = gaussian_pdf(0, 0, 1)
        pdf_away_from_mean = gaussian_pdf(2, 0, 1)

        assert pdf_at_mean > pdf_away_from_mean
        assert pdf_at_mean > 0

    def test_beta_distribution(self):
        """Test beta distribution properties."""
        pytest.importorskip("scipy")
        from scipy.special import gamma

        alpha, beta = 2, 5
        x = np.linspace(0, 1, 100)

        def beta_pdf(x, alpha, beta):
            return (
                (gamma(alpha + beta) / (gamma(alpha) * gamma(beta)))
                * (x ** (alpha - 1))
                * ((1 - x) ** (beta - 1))
            )

        pdf_values = beta_pdf(x, alpha, beta)

        # PDF should be non-negative and integrate to approximately 1
        assert np.all(pdf_values >= 0)
        integration = np.trapezoid(pdf_values, x)
        assert abs(integration - 1.0) < 0.1

    def test_diffusion_process_simulation(self):
        """Test diffusion process simulation."""

        def simulate_diffusion_step(current_pos, dt=0.01, diffusion_coeff=1.0):
            """Simulate one step of Brownian motion."""
            noise = np.random.normal(0, np.sqrt(2 * diffusion_coeff * dt))
            return current_pos + noise

        # Run simulation
        initial_pos = 0.0
        steps = 100
        positions = [initial_pos]

        for _ in range(steps):
            new_pos = simulate_diffusion_step(positions[-1])
            positions.append(new_pos)

        positions = np.array(positions)

        # Note: This is a statistical test, might occasionally fail due to randomness
        # In practice, you'd want to run multiple simulations or use a fixed seed
        assert len(positions) == steps + 1


class TestPlotGeneration:
    """Test that plots can be generated without errors."""

    def test_basic_plot_creation(self):
        """Test basic matplotlib plot generation."""
        fig, ax = plt.subplots(figsize=(8, 6))

        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)

        ax.plot(x, y, label="sin(x)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)

        # Should not raise any errors
        plt.close(fig)

    def test_histogram_creation(self):
        """Test histogram generation."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Generate sample data
        data = np.random.normal(0, 1, 1000)

        ax.hist(data, bins=30, alpha=0.7, density=True, label="Data")

        # Overlay theoretical PDF
        x = np.linspace(-4, 4, 100)
        theoretical_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
        ax.plot(x, theoretical_pdf, "r-", label="Theoretical")

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

        plt.close(fig)

    def test_subplot_creation(self):
        """Test subplot generation for multi-panel plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

        # Different plots in each subplot
        x = np.linspace(0, 10, 100)

        ax1.plot(x, np.sin(x))
        ax1.set_title("Sine")

        ax2.plot(x, np.cos(x))
        ax2.set_title("Cosine")

        ax3.scatter(x[::10], np.sin(x[::10]))
        ax3.set_title("Scatter")

        ax4.bar(range(5), [1, 3, 2, 4, 2])
        ax4.set_title("Bar")

        plt.tight_layout()
        plt.close(fig)


class TestStatisticalFunctions:
    """Test statistical functions commonly used in the project."""

    def test_bias_variance_decomposition_concept(self):
        """Test the mathematical concept of bias-variance decomposition."""

        # Simulate a simple scenario
        def true_function(x):
            return x**2

        # Generate training data with noise
        np.random.seed(42)  # For reproducible tests
        x_train = np.linspace(0, 1, 20)
        y_train = true_function(x_train) + 0.1 * np.random.normal(size=len(x_train))

        # Test point
        x_test = 0.5
        true_value = true_function(x_test)

        # Simulate multiple model predictions (simplified linear models)
        predictions = []
        for _ in range(100):
            # Add noise to training and fit simple model
            noisy_y = y_train + 0.05 * np.random.normal(size=len(y_train))
            # Simple linear fit
            coeffs = np.polyfit(x_train, noisy_y, 1)
            prediction = np.polyval(coeffs, x_test)
            predictions.append(prediction)

        predictions = np.array(predictions)

        # Calculate bias and variance
        mean_prediction = np.mean(predictions)
        bias_squared = (mean_prediction - true_value) ** 2
        variance = np.var(predictions)

        # Basic sanity checks
        assert bias_squared >= 0
        assert variance >= 0
        assert len(predictions) == 100

    def test_bayesian_update_concept(self):
        """Test Bayesian updating concept."""
        # Simple Beta-Binomial example
        # Prior: Beta(1, 1) (uniform)
        prior_alpha, prior_beta = 1, 1

        # Observed data: 7 successes out of 10 trials
        successes, trials = 7, 10
        failures = trials - successes

        # Posterior: Beta(prior_alpha + successes, prior_beta + failures)
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + failures

        # Basic checks
        assert posterior_alpha == 8
        assert posterior_beta == 4

        # Mean of Beta distribution
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        expected_mean = successes / trials  # Should be close to observed proportion

        # With weak prior, posterior should be close to data
        assert abs(posterior_mean - expected_mean) < 0.1
