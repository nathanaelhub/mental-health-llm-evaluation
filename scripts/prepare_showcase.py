#!/usr/bin/env python3
"""
Prepare project for academic showcase
"""

import os
import shutil
from pathlib import Path
import glob

def create_showcase_structure():
    """Create a clean structure for presentation"""
    
    print("🎯 Preparing project for showcase...")
    
    # Create showcase directories
    showcase_dirs = [
        "results/showcase",
        "docs/technical", 
        "docs/guides",
        "scripts/archive"
    ]
    
    for dir_path in showcase_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}")
    
    # Copy best visualizations to showcase (if not already there)
    best_results = "results/development/four_model_sample_20250731_150627/visualizations"
    if Path(best_results).exists():
        viz_files = [
            "1_model_comparison.png",
            "2_detailed_scores_heatmap.png", 
            "3_dimension_radar.png",
            "4_category_performance.png",
            "5_response_times.png",
            "6_summary_infographic.png"
        ]
        
        copied = 0
        for viz in viz_files:
            src = Path(best_results) / viz
            dst = Path("results/showcase") / viz
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
                print(f"✅ Copied {viz} to showcase")
        
        if copied == 0:
            print("✅ Showcase visualizations already present")
    
    # Create .gitignore for sensitive files
    gitignore_path = Path(".gitignore")
    gitignore_additions = """
# Capstone private notes
docs/capstone_notes/

# Development artifacts
*.log
*.pyc
__pycache__/

# Local configuration
.env
*.db
conversations.db

# Temporary files
temp/
cache/

# API keys and secrets
*_api_key*
*_secret*

# Development test files
scripts/test_*.py
scripts/debug_*.py
scripts/demo_*.py
"""
    
    # Check if additions already exist
    existing_content = ""
    if gitignore_path.exists():
        existing_content = gitignore_path.read_text()
    
    if "# Capstone private notes" not in existing_content:
        with open(gitignore_path, "a") as f:
            f.write(gitignore_additions)
        print("✅ Updated .gitignore")
    else:
        print("✅ .gitignore already configured")
    
    # Create LICENSE file
    license_path = Path("LICENSE")
    if not license_path.exists():
        mit_license = """MIT License

Copyright (c) 2025 Mental Health LLM Evaluation Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        
        license_path.write_text(mit_license)
        print("✅ Created LICENSE file")
    else:
        print("✅ LICENSE file already exists")
    
    # Archive development/test files
    scripts_dir = Path("scripts")
    archive_dir = scripts_dir / "archive"
    
    dev_patterns = [
        "test_*.py",
        "debug_*.py", 
        "demo_*.py"
    ]
    
    archived = 0
    for pattern in dev_patterns:
        for file in scripts_dir.glob(pattern):
            if file.is_file() and file.name not in ["test_local_response_times.py"]:  # Keep current test
                dst = archive_dir / file.name
                if not dst.exists():
                    shutil.move(str(file), str(dst))
                    archived += 1
                    print(f"✅ Archived {file.name}")
    
    if archived == 0:
        print("✅ No development files to archive")
    
    # Clean up root directory
    root_cleanup_patterns = [
        "*.log",
        "*.tmp", 
        "*_backup.*",
        "*_old.*"
    ]
    
    cleaned = 0
    for pattern in root_cleanup_patterns:
        for file in Path(".").glob(pattern):
            if file.is_file():
                file.unlink()
                cleaned += 1
                print(f"✅ Cleaned up {file.name}")
    
    if cleaned == 0:
        print("✅ Root directory already clean")
    
    # Create showcase summary
    showcase_summary = """# Showcase Files Summary

## 📊 Key Presentation Materials

### Primary Visualizations (results/showcase/)
- `1_model_comparison.png` - Overall performance comparison
- `2_detailed_scores_heatmap.png` - Multi-dimensional analysis  
- `3_dimension_radar.png` - Therapeutic quality radar
- `4_category_performance.png` - Mental health category performance
- `5_response_times.png` - Speed and efficiency analysis
- `6_summary_infographic.png` - Complete research summary

### Essential Documentation
- `EXECUTIVE_SUMMARY.md` - One-page project overview
- `docs/RESEARCH_FINDINGS.md` - Key results and insights
- `docs/DEMO_GUIDE.md` - Presentation demonstration guide
- `results/showcase/key_findings.md` - Statistical highlights

### Technical Reference
- `docs/technical/SYSTEM_ARCHITECTURE.md` - Complete system design
- `docs/technical/API_DOCUMENTATION.md` - Implementation details

## 🎯 Presentation Flow Recommendation

1. **Executive Summary** (2 minutes)
2. **Live Demo** using DEMO_GUIDE.md (5 minutes)  
3. **Research Results** with showcase visualizations (8 minutes)
4. **Technical Architecture** overview (3 minutes)
5. **Q&A** using research findings (2 minutes)

Total: 20 minutes with professional materials
"""
    
    showcase_readme = Path("results/showcase/README.md")
    if not showcase_readme.exists():
        showcase_readme.write_text(showcase_summary)
        print("✅ Created showcase README")
    
    print("\n🎉 Project showcase preparation complete!")
    print("\n📋 Showcase Status:")
    print("✅ Professional documentation structure")
    print("✅ Clean presentation materials")
    print("✅ Archive development files")
    print("✅ Academic-ready repository")
    print("\n🚀 Next steps:")
    print("1. Review README.md for any personal information")
    print("2. Test demo using docs/DEMO_GUIDE.md")
    print("3. Commit changes: git add -A && git commit -m 'Prepare for academic showcase'")
    print("4. Create public GitHub repository for portfolio")

if __name__ == "__main__":
    create_showcase_structure()