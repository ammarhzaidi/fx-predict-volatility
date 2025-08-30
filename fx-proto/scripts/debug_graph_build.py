# scripts/debug_graph_build.py
"""
Debug version to understand why graph_build.py doesn't work
"""

import sys
from pathlib import Path

# Setup paths
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))


def test_imports():
    """Test all imports from graph_build.py"""
    print("🔍 Testing imports from graph_build.py...")

    try:
        import numpy as np
        print("   ✅ numpy")
    except ImportError as e:
        print(f"   ❌ numpy: {e}")
        return False

    try:
        import pandas as pd
        print("   ✅ pandas")
    except ImportError as e:
        print(f"   ❌ pandas: {e}")
        return False

    try:
        import networkx as nx
        print("   ✅ networkx")
    except ImportError as e:
        print(f"   ❌ networkx: {e}")
        return False

    try:
        import matplotlib.pyplot as plt
        print("   ✅ matplotlib")
    except ImportError as e:
        print(f"   ❌ matplotlib: {e}")
        return False

    try:
        import seaborn as sns
        print("   ✅ seaborn")
    except ImportError as e:
        print(f"   ❌ seaborn: {e}")
        return False

    try:
        import torch
        import torch.nn as nn
        print("   ✅ torch")
    except ImportError as e:
        print(f"   ❌ torch: {e}")
        return False

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        print("   ✅ sklearn")
    except ImportError as e:
        print(f"   ❌ sklearn: {e}")
        return False

    return True


def test_graph_build_classes():
    """Test importing classes from graph_build.py"""
    print("\n🔍 Testing graph_build.py classes...")

    try:
        from fxproto.graphdemo.graph_build import FinancialKnowledgeGraph
        print("   ✅ FinancialKnowledgeGraph imported")

        # Try to create an instance
        kg = FinancialKnowledgeGraph()
        print(f"   ✅ Knowledge graph created with {len(kg.graph.nodes())} nodes")

    except Exception as e:
        print(f"   ❌ FinancialKnowledgeGraph failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        from fxproto.graphdemo.graph_build import GraphAttentionLayer
        print("   ✅ GraphAttentionLayer imported")
    except Exception as e:
        print(f"   ❌ GraphAttentionLayer failed: {e}")
        return False

    try:
        from fxproto.graphdemo.graph_build import GraphForexPredictor
        print("   ✅ GraphForexPredictor imported")
    except Exception as e:
        print(f"   ❌ GraphForexPredictor failed: {e}")
        return False

    return True


def test_main_demo():
    """Test the main_demo function"""
    print("\n🔍 Testing main_demo function...")

    try:
        from fxproto.graphdemo.graph_build import main_demo
        print("   ✅ main_demo function imported")

        # Try to call it
        print("   🚀 Calling main_demo()...")
        result = main_demo()

        if result:
            print("   ✅ main_demo() executed successfully")
            return True
        else:
            print("   ❌ main_demo() returned None or False")
            return False

    except Exception as e:
        print(f"   ❌ main_demo() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_script_execution():
    """Check if the script would execute when run directly"""
    print("\n🔍 Checking script execution logic...")

    # Load the script and check the __name__ == "__main__" block
    script_path = src_dir / "fxproto" / "graphdemo" / "graph_build.py"

    if not script_path.exists():
        print(f"   ❌ Script not found: {script_path}")
        return False

    print(f"   ✅ Script found: {script_path}")

    # Read the script content
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if it has the main execution block
    if 'if __name__ == "__main__":' in content:
        print("   ✅ Has main execution block")

        # Check what's in the main block
        lines = content.split('\n')
        in_main = False
        main_content = []

        for line in lines:
            if 'if __name__ == "__main__":' in line:
                in_main = True
                continue
            elif in_main and line.startswith(' ') or line.startswith('\t'):
                main_content.append(line.strip())
            elif in_main and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                break

        print("   📋 Main block content:")
        for line in main_content:
            if line:
                print(f"      {line}")

        if not main_content or all(not line for line in main_content):
            print("   ⚠️  Main block is empty!")
            return False

    else:
        print("   ❌ No main execution block found")
        return False

    return True


def main():
    """Run all debug tests"""
    print("🐛 DEBUG: Graph Build Script Analysis")
    print("=" * 50)

    # Test 1: Basic imports
    imports_ok = test_imports()

    if not imports_ok:
        print("❌ Basic imports failed - cannot continue")
        return

    # Test 2: Graph build classes
    classes_ok = test_graph_build_classes()

    # Test 3: Script execution logic
    execution_ok = check_script_execution()

    # Test 4: Try to run main_demo
    if classes_ok:
        demo_ok = test_main_demo()
    else:
        demo_ok = False

    # Summary
    print("\n" + "=" * 50)
    print("🐛 DEBUG SUMMARY")
    print("=" * 50)

    tests = [
        ("Basic Imports", imports_ok),
        ("Graph Classes", classes_ok),
        ("Script Execution Logic", execution_ok),
        ("Main Demo Function", demo_ok)
    ]

    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")

    if not demo_ok:
        print(f"\n💡 RECOMMENDATION:")
        print(f"   The original graph_build.py might have issues.")
        print(f"   Try running: python scripts/simple_graph_demo.py")
        print(f"   This is a working version I created for you.")


if __name__ == "__main__":
    main()