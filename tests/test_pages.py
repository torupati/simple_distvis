"""Test page imports and basic functionality."""

import warnings

import pytest

# Filter out Streamlit warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")


class TestPageImports:
    """Test that all remaining pages can be imported without errors."""

    def test_diffusion_process_import(self):
        """Test that diffusion_process.py can be imported."""
        try:
            import diffusion_process

            assert hasattr(diffusion_process, "st")
        except ImportError as e:
            pytest.fail(f"Failed to import diffusion_process: {e}")

    def test_beta_bayes_import(self):
        """Test that beta_bayes.py can be imported."""
        try:
            import beta_bayes

            assert hasattr(beta_bayes, "st")
        except ImportError as e:
            pytest.fail(f"Failed to import beta_bayes: {e}")

    def test_normal_distribution1d_import(self):
        """Test that normal_distribution1d.py can be imported."""
        try:
            import normal_distribution1d

            assert hasattr(normal_distribution1d, "st")
        except ImportError as e:
            pytest.fail(f"Failed to import normal_distribution1d: {e}")

    def test_gaussian1d_posteriori_import(self):
        """Test that 03_Gaussian1D_posteriori.py can be imported."""
        try:
            # Import with module name that starts with number
            spec = pytest.importorskip("importlib.util")
            import importlib.util
            from pathlib import Path

            pages_dir = Path(__file__).parent.parent / "pages"
            gaussian_path = pages_dir / "03_Gaussian1D_posteriori.py"

            spec = importlib.util.spec_from_file_location(
                "gaussian1d_posteriori", gaussian_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            assert hasattr(module, "st")
        except Exception as e:
            pytest.fail(f"Failed to import 03_Gaussian1D_posteriori: {e}")

    def test_square_error_decomposition_import(self):
        """Test that square_error_decomposition.py can be imported."""
        try:
            import square_error_decomposition

            assert hasattr(square_error_decomposition, "st")
        except ImportError as e:
            pytest.fail(f"Failed to import square_error_decomposition: {e}")

    def test_square_error_decomposition_2d_import(self):
        """Test that square_error_decomposition_2d.py can be imported."""
        try:
            import square_error_decomposition_2d

            assert hasattr(square_error_decomposition_2d, "st")
        except ImportError as e:
            pytest.fail(f"Failed to import square_error_decomposition_2d: {e}")


class TestNoHMMImports:
    """Test that HMM-related imports are completely removed."""

    def test_no_hmm_imports_in_pages(self):
        """Ensure no pages try to import from src.hmm."""
        import re
        from pathlib import Path

        pages_dir = Path(__file__).parent.parent / "pages"
        hmm_import_pattern = re.compile(r"from\s+src\.hmm|import.*hmm|src\.hmm")

        for page_file in pages_dir.glob("*.py"):
            with open(page_file, "r", encoding="utf-8") as f:
                content = f.read()
                matches = hmm_import_pattern.findall(content)
                assert not matches, f"Found HMM imports in {page_file.name}: {matches}"

    def test_no_hmm_directory(self):
        """Ensure src/hmm directory no longer exists."""
        from pathlib import Path

        hmm_dir = Path(__file__).parent.parent / "src" / "hmm"
        assert not hmm_dir.exists(), "src/hmm directory should not exist"


class TestProjectDependencies:
    """Test that project dependencies are correctly configured."""

    def test_required_packages_importable(self):
        """Test that all required packages can be imported."""
        required_packages = ["numpy", "matplotlib", "streamlit"]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package {package} is not importable")

    def test_removed_dependencies_not_required(self):
        """Test that removed dependencies are not accidentally used."""
        # These packages were removed when HMM functionality was deleted
        removed_packages = ["sklearn", "scikit-learn", "tqdm"]

        # We don't test that they can't be imported (they might be installed)
        # but we verify they're not in our requirements
        from pathlib import Path

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()

        for package in removed_packages:
            assert package not in content, (
                f"Removed package {package} found in pyproject.toml"
            )
