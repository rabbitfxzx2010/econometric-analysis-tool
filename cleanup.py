#!/usr/bin/env python3
"""
Cleanup script: Remove temporary test files and reset app state
"""
import os
import glob

# Files to clean up
cleanup_files = [
    "test_app.py",
    "app_usage_stats.json",
    "user_feedback.txt",
    "email_count_*.txt",
    "__pycache__"
]

print("🧹 Cleaning up temporary files...")

for pattern in cleanup_files:
    if "*" in pattern:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                print(f"   Removed: {file}")
            except Exception as e:
                print(f"   Could not remove {file}: {e}")
    else:
        if os.path.exists(pattern):
            try:
                if os.path.isdir(pattern):
                    import shutil
                    shutil.rmtree(pattern)
                else:
                    os.remove(pattern)
                print(f"   Removed: {pattern}")
            except Exception as e:
                print(f"   Could not remove {pattern}: {e}")

print("\n✅ Cleanup complete!")
print("\n📁 Remaining files for production:")
os.system("ls -la | grep -E '\\.py$|\\.csv$|\\.md$|requirements.txt'")
