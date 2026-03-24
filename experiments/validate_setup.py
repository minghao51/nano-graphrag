#!/usr/bin/env python3
"""Validate that the benchmark environment is properly configured."""

import sys
from pathlib import Path


def check_config_file(config_path):
    """Validate a single config file."""
    print(f"Checking {config_path}...", end=" ")

    if not Path(config_path).exists():
        print("❌ MISSING")
        return False

    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required fields
        required_fields = ["name", "dataset", "graphrag", "query", "metrics", "output"]
        for field in required_fields:
            if field not in config:
                print(f"❌ MISSING FIELD: {field}")
                return False

        print("✅")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_dependencies():
    """Check required Python packages."""
    print("\n📦 Checking dependencies...")

    required = [
        ("yaml", "pyyaml"),
        ("datasets", "datasets"),
        ("nano_graphrag", "nano-graphrag"),
    ]

    all_ok = True
    for module, package in required:
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (install with: uv add {package})")
            all_ok = False

    return all_ok


def check_directories():
    """Check/create required directories."""
    print("\n📁 Checking directories...")

    dirs = [
        "./experiments",
        "./results",
        "./workdirs",
        "~/.cache/nano-bench",
    ]

    for dir_path in dirs:
        dir_path = Path(dir_path).expanduser()
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created {dir_path}")
            except Exception as e:
                print(f"  ❌ Cannot create {dir_path}: {e}")
                return False
        else:
            print(f"  ✅ {dir_path}")

    return True


def check_configs():
    """Check all benchmark configs."""
    print("\n📄 Checking benchmark configs...")

    configs = [
        "experiments/benchmark_multihop_rag.yaml",
        "experiments/benchmark_musique.yaml",
        "experiments/benchmark_hotpotqa.yaml",
        "experiments/benchmark_2wiki.yaml",
    ]

    all_ok = True
    for config in configs:
        if not check_config_file(config):
            all_ok = False

    return all_ok


def check_api_keys():
    """Check for API keys (optional)."""
    print("\n🔑 Checking API keys...")

    # Check for OpenAI API key
    if Path(".env").exists():
        print("  ✅ .env file found")
        with open(".env", "r") as f:
            if "OPENAI_API_KEY" in f.read():
                print("  ✅ OPENAI_API_KEY found")
            else:
                print("  ⚠️  OPENAI_API_KEY not in .env")
    else:
        print("  ⚠️  No .env file found (optional if using other providers)")

    return True  # Not required


def main():
    print("=" * 60)
    print("BENCHMARK SETUP VALIDATION")
    print("=" * 60)

    checks = [
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Configs", check_configs),
        ("API Keys", check_api_keys),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error checking {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:<10} {name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✅ All checks passed! Ready to run benchmarks.")
        print("\nNext steps:")
        print("  1. Quick test: ./run_all_benchmarks.sh --quick")
        print("  2. Larger run: edit max_samples/max_corpus_samples in the YAMLs, then rerun")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
