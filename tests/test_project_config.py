"""Test project configuration and structure."""

from pathlib import Path

import toml


class TestProjectStructure:
    """Test project structure and configuration."""

    def test_pyproject_toml_structure(self):
        """Test that pyproject.toml has correct structure."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        assert pyproject_path.exists(), "pyproject.toml should exist"

        with open(pyproject_path, "r", encoding="utf-8") as f:
            config = toml.load(f)

        # Check required sections
        assert "project" in config
        assert "build-system" in config

        # Check project metadata
        project = config["project"]
        assert "name" in project
        assert "version" in project
        assert "dependencies" in project

        # Check required dependencies are present
        dependencies = project["dependencies"]
        required_deps = ["matplotlib", "streamlit", "numpy"]

        for dep in required_deps:
            assert any(dep in str(d).lower() for d in dependencies), (
                f"Missing dependency: {dep}"
            )

        # Check removed dependencies are not present
        removed_deps = ["scikit-learn", "sklearn", "tqdm"]
        for dep in removed_deps:
            assert not any(dep in str(d).lower() for d in dependencies), (
                f"Should not have dependency: {dep}"
            )

    def test_pages_directory_structure(self):
        """Test pages directory structure."""
        project_root = Path(__file__).parent.parent
        pages_dir = project_root / "pages"

        assert pages_dir.exists(), "pages directory should exist"

        # Expected pages (no HMM-related pages)
        expected_pages = [
            "03_Gaussian1D_posteriori.py",
            "beta_bayes.py",
            "diffusion_process.py",
            "normal_distribution1d.py",
            "square_error_decomposition.py",
            "square_error_decomposition_2d.py",
        ]

        existing_pages = [p.name for p in pages_dir.glob("*.py")]

        for page in expected_pages:
            assert page in existing_pages, f"Expected page {page} not found"

        # Check that no HMM-related pages exist
        hmm_related_patterns = [
            "hmm",
            "gmm",
            "kmeans",
            "markov",
            "sample_generator",
            "clustering",
        ]
        for page in existing_pages:
            for pattern in hmm_related_patterns:
                assert pattern not in page.lower(), f"HMM-related page found: {page}"

    def test_src_directory_structure(self):
        """Test src directory structure."""
        project_root = Path(__file__).parent.parent
        src_dir = project_root / "src"

        assert src_dir.exists(), "src directory should exist"
        assert (src_dir / "__init__.py").exists(), "src/__init__.py should exist"

        # Ensure HMM directory is completely removed
        hmm_dir = src_dir / "hmm"
        assert not hmm_dir.exists(), "src/hmm directory should not exist"

    def test_tests_directory_structure(self):
        """Test tests directory structure."""
        project_root = Path(__file__).parent.parent
        tests_dir = project_root / "tests"

        assert tests_dir.exists(), "tests directory should exist"
        assert (tests_dir / "conftest.py").exists(), "tests/conftest.py should exist"

        # Check no HMM test files exist
        test_files = list(tests_dir.glob("*.py"))
        for test_file in test_files:
            assert "hmm" not in test_file.name.lower(), (
                f"HMM test file should not exist: {test_file.name}"
            )

    def test_docker_configuration(self):
        """Test Docker configuration exists."""
        project_root = Path(__file__).parent.parent
        docker_dir = project_root / "docker"

        assert docker_dir.exists(), "docker directory should exist"
        assert (docker_dir / "Dockerfile").exists(), "Dockerfile should exist"

    def test_documentation_structure(self):
        """Test documentation structure."""
        project_root = Path(__file__).parent.parent
        docs_dir = project_root / "docs"

        assert docs_dir.exists(), "docs directory should exist"
        assert (docs_dir / "conf.py").exists(), "docs/conf.py should exist"
        assert (docs_dir / "index.rst").exists(), "docs/index.rst should exist"

        # Check that HMM documentation is removed
        hmm_rst = docs_dir / "src.hmm.rst"
        assert not hmm_rst.exists(), "src.hmm.rst should not exist"


class TestApplicationConfiguration:
    """Test application configuration."""

    def test_app_py_exists(self):
        """Test that main app.py exists and has correct structure."""
        project_root = Path(__file__).parent.parent
        app_py = project_root / "app.py"

        assert app_py.exists(), "app.py should exist"

        with open(app_py, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that it imports streamlit
        assert "import streamlit" in content, "app.py should import streamlit"

        # Check that it sets up page config
        assert "st.set_page_config" in content, "app.py should set page config"

        # Check that HMM-related content is removed from description
        hmm_terms = ["hmm", "k-means", "clustering", "gmm", "gaussian mixture"]
        for term in hmm_terms:
            assert term.lower() not in content.lower(), (
                f"HMM-related term '{term}' found in app.py"
            )

    def test_requirements_consistency(self):
        """Test that requirements.txt is consistent with pyproject.toml."""
        project_root = Path(__file__).parent.parent

        # Read pyproject.toml dependencies
        pyproject_path = project_root / "pyproject.toml"
        with open(pyproject_path, "r", encoding="utf-8") as f:
            config = toml.load(f)

        pyproject_deps = config["project"]["dependencies"]

        # Check if requirements.txt exists
        requirements_path = project_root / "requirements.txt"
        if requirements_path.exists():
            with open(requirements_path, "r", encoding="utf-8") as f:
                requirements_content = f.read()

            # Basic consistency check - main packages should be present
            main_packages = ["streamlit", "numpy", "matplotlib"]
            for package in main_packages:
                found_in_pyproject = any(
                    package in str(dep).lower() for dep in pyproject_deps
                )
                found_in_requirements = package.lower() in requirements_content.lower()

                # If it's in pyproject, it should be in requirements (if requirements exists)
                if found_in_pyproject:
                    assert found_in_requirements or not requirements_path.exists(), (
                        f"Package {package} in pyproject.toml but not in requirements.txt"
                    )
