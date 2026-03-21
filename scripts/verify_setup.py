#!/usr/bin/env python
"""Verify environment setup for the thesis project.

Usage:
    python scripts/verify_setup.py          # Basic environment check
    python scripts/verify_setup.py --all    # Full check (env + sqlite + pipeline)
"""
import argparse
import subprocess
import sys
from pathlib import Path


def check_import(module_name, friendly_name=None):
    try:
        __import__(module_name)
        name = friendly_name or module_name
        print(f"✓ {name}")
        return True
    except ImportError as e:
        name = friendly_name or module_name
        print(f"✗ {name}: {e}")
        return False


def check_project_imports():
    """Check that project modules can be imported."""
    print("\n7. Project modules:")

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    modules = [
        ("src.utils.config", "UCaGNNConfig"),
        ("src.utils.experiment_logger", "ExperimentLogger"),
        ("src.models.ucagnn", "UCaGNN"),
        ("src.losses.loss_suite", "LossSuite"),
        ("src.training.trainer", "Trainer"),
        ("src.data.loaders", "load_dataset"),
        ("src.data.graph_builder", "build_graph"),
        ("src.profiling.gpu_profiler", "GPUProfiler"),
    ]

    all_good = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name} ({module})")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            all_good = False

    return all_good


def main():
    parser = argparse.ArgumentParser(description="Verify thesis environment setup")
    parser.add_argument("--all", action="store_true", help="Run full verification (env + sqlite + pipeline)")
    args = parser.parse_args()

    print("=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)

    print("\n1. PyTorch:")
    all_good = all([
        check_import("torch", "PyTorch"),
    ])

    print("\n2. PyTorch-Geometric:")
    all_good &= all([
        check_import("torch_geometric", "PyTorch Geometric"),
    ])

    print("\n3. MLFlow:")
    all_good &= all([
        check_import("mlflow", "MLFlow"),
    ])

    print("\n4. Data libraries:")
    all_good &= all([
        check_import("pandas", "Pandas"),
        check_import("numpy", "NumPy"),
        check_import("scipy", "SciPy"),
        check_import("sklearn", "scikit-learn"),
        check_import("polars", "Polars"),
    ])

    print("\n5. Visualization:")
    all_good &= check_import("matplotlib", "Matplotlib")

    print("\n6. Dev tools:")
    all_good &= check_import("ruff", "ruff (formatter)")

    print("\n7. GPU test:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ CUDA available / GPU: {gpu_name} ({vram:.1f} GB)")
        else:
            print("⚠ CUDA not available (CPU mode)")
            # Don't fail -- CPU mode is valid for development
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        all_good = False

    all_good &= check_project_imports()

    print("\n" + "=" * 60)
    if all_good:
        print("✓ ENVIRONMENT CHECKS PASSED")
    else:
        print("✗ SOME ENVIRONMENT CHECKS FAILED")
        sys.exit(1)

    # If --all, run additional verification scripts
    if args.all:
        scripts_dir = Path(__file__).parent

        print("\n")
        print("=" * 60)
        print("RUNNING SQLITE VERIFICATION...")
        print("=" * 60)
        result = subprocess.run([sys.executable, scripts_dir / "verify_sqlite.py"])
        if result.returncode != 0:
            sys.exit(1)

        print("\n")
        print("=" * 60)
        print("RUNNING PIPELINE SANITY CHECK...")
        print("=" * 60)
        result = subprocess.run([sys.executable, scripts_dir / "verify_pipeline.py"])
        if result.returncode != 0:
            sys.exit(1)

        print("\n" + "=" * 60)
        print("✓ ALL VERIFICATION CHECKS PASSED")
        print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
