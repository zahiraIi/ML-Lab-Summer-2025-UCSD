#!/usr/bin/env python3
"""
Clean run script for navigation testing
Removes unnecessary files and runs the core navigation test
"""

import os
import shutil
from pathlib import Path

def cleanup_unnecessary_files():
    """Remove unnecessary files, keep only what's needed"""
    print("🧹 Cleaning up unnecessary files...")
    
    # Files to keep
    keep_files = {
        'mpc_triangle_alignment.ipynb',
        'ddpg_3bot_line_training.ipynb',
        'multibot_cluster_env.py',
        'enhanced_multibot_env.py',
        'step3_decentralized_swarm.py',
        'run_navigation_test.py',
        'run_clean_navigation_test.py'
    }
    
    # Directories to keep
    keep_dirs = {
        'ddpg_tensorboard',
        '__pycache__',
        '.ipynb_checkpoints',
        '.vscode',
        'navigation_results'
    }
    
    # Get all files and directories
    current_dir = Path('.')
    all_items = list(current_dir.iterdir())
    
    removed_count = 0
    for item in all_items:
        if item.name not in keep_files and item.name not in keep_dirs:
            try:
                if item.is_file():
                    item.unlink()
                    print(f"   ❌ Removed file: {item.name}")
                    removed_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    print(f"   ❌ Removed directory: {item.name}")
                    removed_count += 1
            except Exception as e:
                print(f"   ⚠️ Could not remove {item.name}: {e}")
    
    print(f"✅ Cleanup completed! Removed {removed_count} items.")
    print("\n📁 Kept essential files:")
    for file in sorted(keep_files):
        if Path(file).exists():
            print(f"   ✓ {file}")

def main():
    """Main function"""
    print("🚀 Clean Navigation Test System")
    print("=" * 50)
    
    # Cleanup first
    cleanup_unnecessary_files()
    
    print("\n🤖 Starting navigation test...")
    print("This will test RL algorithms on point A → point B navigation")
    print("using the exact physics from the notebooks.")
    print()
    
    # Import and run after cleanup
    try:
        from run_navigation_test import NavigationTestSystem
        
        tester = NavigationTestSystem()
        results = tester.run_quick_comparison()
        
        print("\n✅ Navigation test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure run_navigation_test.py is in the current directory")
    except Exception as e:
        print(f"❌ Error running test: {e}")

if __name__ == "__main__":
    main() 