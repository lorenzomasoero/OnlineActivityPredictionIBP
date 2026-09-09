from pathlib import Path


def test_source_tree_contains_only_scoped_inputs_and_no_generated_artifacts():
    root = Path(__file__).parents[1]
    # Ignore generated/output directories (gitignored; not part of the source tree).
    IGNORED = {"out", "output", "debug", "__pycache__", ".venv", ".pytest_cache",
               ".git", "sweep", "values"}
    relative_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not any(part in IGNORED for part in path.relative_to(root).parts)
    }
    assert not any("mini_code" in path for path in relative_files)
    assert not any("beraha@" in path for path in relative_files)
    assert not any("amazon_data" in path for path in relative_files)
    assert not any(
        path.endswith((".pdf", ".ipynb", ".pickle")) for path in relative_files
    )
    assert all(
        path.startswith("data/")
        for path in relative_files
        if path.endswith(".npy")
    )


def test_package_and_scripts_follow_the_experiment_layout():
    root = Path(__file__).parents[1]
    assert (root / "src" / "activity_prediction").is_dir()

    expected_scripts = {
        "run_asos.py",
        "run_hitting_times.py",
        "run_interval_comparison.py",
        "run_inversion.py",
        "run_model_comparison.py",
        "run_nb_prediction.py",
        "run_parameter_estimation.py",
        "run_parameter_sensitivity.py",
        "run_rees46.py",
        "run_uci.py",
        "run_zipf.py",
    }
    assert {path.name for path in (root / "scripts").glob("*.py")} == expected_scripts
