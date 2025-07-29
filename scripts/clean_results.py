#!/usr/bin/env python3
"""
Results Directory Cleanup Tool
=============================

Organizes the results directory by backing up essential files and cleaning up temporary/development files.

Usage:
    python scripts/clean_results.py --dry-run     # Preview what will be deleted
    python scripts/clean_results.py --backup     # Create presentation backup only
    python scripts/clean_results.py --clean      # Remove temporary files only
    python scripts/clean_results.py --full       # Backup + clean (recommended)

File Categories:
    ESSENTIAL: Main research results, final visualizations, statistical analysis
    TEMPORARY: Development outputs, raw conversations, intermediate calculations
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Essential files for research presentation (KEEP THESE)
ESSENTIAL_PATTERNS = [
    "detailed_results.json",              # Main research summary (root level)
    "statistical_analysis.json",          # Statistical analysis (root level)  
    "research_report.txt",                # Research report (root level)
    "visualizations/charts/*.png",        # Final publication charts
    "README.md"                           # Documentation
]

# Temporary/development files (SAFE TO DELETE)
TEMPORARY_PATTERNS = [
    "development/",                       # Entire development folder
    "comparison_*.txt",                   # Quick comparison outputs (all ages)
    "model_strengths.json",               # Redundant (info in statistical_analysis.json)
    "visualizations/visualization_summary.json",  # Redundant metadata
    "visualizations/evaluation_summary.txt",      # Debug output file
    "visualizations/presentation/",                # Legacy slides directory
    "visualizations/.gitkeep",                     # Git placeholder files
]

# Demo files (SPECIAL HANDLING)
DEMO_PATTERNS = [
    "demo/demo_*/*",                      # All files in demo run directories
]

# Directory structure
RESULTS_DIR = Path("results")
BACKUP_DIR = RESULTS_DIR / "backup"

class ResultsCleaner:
    """Handles cleanup and organization of results directory."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            'essential_files': 0,
            'temporary_files': 0,
            'bytes_saved': 0,
            'backed_up': 0
        }
    
    def get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        try:
            return file_path.stat().st_size
        except (OSError, FileNotFoundError):
            return 0
    
    def find_files_by_patterns(self, patterns: List[str], base_dir: Path) -> List[Path]:
        """Find files matching glob patterns."""
        files = []
        for pattern in patterns:
            if pattern.endswith('/'):
                # Directory pattern
                dir_path = base_dir / pattern.rstrip('/')
                if dir_path.exists() and dir_path.is_dir():
                    files.extend(dir_path.rglob('*'))
            else:
                # File pattern
                files.extend(base_dir.glob(pattern))
        return list(set(files))  # Remove duplicates
    
    def analyze_directory(self) -> Dict[str, List[Path]]:
        """Analyze the results directory and categorize files."""
        if not RESULTS_DIR.exists():
            print(f"❌ Results directory not found: {RESULTS_DIR}")
            return {'essential': [], 'temporary': [], 'demo': [], 'unknown': []}
        
        # Find essential files
        essential_files = self.find_files_by_patterns(ESSENTIAL_PATTERNS, RESULTS_DIR)
        essential_files = [f for f in essential_files if f.is_file()]
        
        # Find temporary files
        temporary_files = self.find_files_by_patterns(TEMPORARY_PATTERNS, RESULTS_DIR)
        
        # Find demo files
        demo_files = self.find_files_by_patterns(DEMO_PATTERNS, RESULTS_DIR)
        
        # All files in results
        all_files = list(RESULTS_DIR.rglob('*'))
        all_files = [f for f in all_files if f.is_file() and not f.name.startswith('.')]
        
        # Categorize unknown files
        essential_set = set(essential_files)
        temporary_set = set(temporary_files)
        demo_set = set(demo_files)
        unknown_files = [f for f in all_files if f not in essential_set and f not in temporary_set and f not in demo_set]
        
        return {
            'essential': essential_files,
            'temporary': [f for f in temporary_files if f.is_file()],
            'demo': [f for f in demo_files if f.is_file()],
            'unknown': unknown_files
        }
    
    def create_backup(self, essential_files: List[Path]) -> bool:
        """Create backup of essential files in backup folder."""
        if not essential_files:
            print("⚠️  No essential files found to backup")
            return False
        
        # Create timestamped backup directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = BACKUP_DIR / f"backup_{timestamp}"
        
        print(f"\n📁 Creating backup...")
        
        if self.dry_run:
            print(f"[DRY RUN] Would create backup directory: {backup_dir}")
            for file in essential_files:
                rel_path = file.relative_to(RESULTS_DIR)
                backup_path = backup_dir / rel_path
                print(f"[DRY RUN] Would copy: {file} → {backup_path}")
            return True
        
        # Create backup directory
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy essential files maintaining structure
        copied_count = 0
        for file in essential_files:
            try:
                rel_path = file.relative_to(RESULTS_DIR)
                backup_path = backup_dir / rel_path
                
                # Create parent directories
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(file, backup_path)
                copied_count += 1
                self.stats['backed_up'] += 1
                
                print(f"✅ Backed up: {rel_path}")
                
            except Exception as e:
                print(f"❌ Failed to backup {file}: {e}")
        
        print(f"\n✅ Backup complete: {copied_count} files copied to {backup_dir}")
        return True
    
    def clean_temporary(self, temporary_files: List[Path]) -> bool:
        """Remove temporary files and directories."""
        if not temporary_files:
            print("ℹ️  No temporary files found to clean")
            return True
        
        print(f"\n🧹 Cleaning temporary files...")
        
        # Group by directory for better organization
        directories_to_remove = set()
        files_to_remove = []
        
        for file in temporary_files:
            if file.is_file():
                files_to_remove.append(file)
            elif file.is_dir():
                directories_to_remove.add(file)
        
        # Remove directories
        for dir_path in sorted(directories_to_remove):
            if self.dry_run:
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                print(f"[DRY RUN] Would remove directory: {dir_path} ({size:,} bytes)")
                self.stats['bytes_saved'] += size
            else:
                try:
                    size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    shutil.rmtree(dir_path)
                    print(f"🗑️  Removed directory: {dir_path} ({size:,} bytes)")
                    self.stats['bytes_saved'] += size
                    self.stats['temporary_files'] += 1
                except Exception as e:
                    print(f"❌ Failed to remove {dir_path}: {e}")
        
        # Remove individual files
        for file in files_to_remove:
            if file.exists():  # Check if not already removed by directory deletion
                size = self.get_file_size(file)
                if self.dry_run:
                    print(f"[DRY RUN] Would remove: {file} ({size:,} bytes)")
                    self.stats['bytes_saved'] += size
                else:
                    try:
                        file.unlink()
                        print(f"🗑️  Removed: {file} ({size:,} bytes)")
                        self.stats['bytes_saved'] += size
                        self.stats['temporary_files'] += 1
                    except Exception as e:
                        print(f"❌ Failed to remove {file}: {e}")
        
        return True
    
    def clean_demo_files(self, demo_files: List[Path], keep_latest: bool = False) -> bool:
        """Clean demo files, optionally keeping only the latest."""
        if not demo_files:
            print("ℹ️  No demo files found to clean")
            return True
            
        # Group demo files by directory (demo run)
        demo_dirs = {}
        for file in demo_files:
            if file.parent.name.startswith('demo_'):
                demo_dir = file.parent
                if demo_dir not in demo_dirs:
                    demo_dirs[demo_dir] = []
                demo_dirs[demo_dir].append(file)
        
        if keep_latest and demo_dirs:
            # Find the latest demo directory by name (assumes timestamp format)
            latest_demo_dir = max(demo_dirs.keys(), key=lambda d: d.name)
            print(f"\n🎬 Managing demo files (keeping latest: {latest_demo_dir.name})...")
            
            for demo_dir, files in demo_dirs.items():
                if demo_dir == latest_demo_dir:
                    print(f"   ✅ Keeping: {demo_dir.relative_to(RESULTS_DIR)}")
                else:
                    print(f"   🗑️  Removing: {demo_dir.relative_to(RESULTS_DIR)}")
                    if self.dry_run:
                        for file in files:
                            self.stats['bytes_saved'] += self.get_file_size(file)
                        self.stats['temporary_files'] += len(files)
                    else:
                        try:
                            shutil.rmtree(demo_dir)
                            self.stats['temporary_files'] += len(files)
                            for file in files:
                                self.stats['bytes_saved'] += self.get_file_size(file)
                        except Exception as e:
                            print(f"   ❌ Error removing {demo_dir}: {e}")
                            return False
        else:
            # Remove all demo files
            print(f"\n🎬 Removing all demo files...")
            for demo_dir, files in demo_dirs.items():
                print(f"   🗑️  Removing: {demo_dir.relative_to(RESULTS_DIR)}")
                if self.dry_run:
                    for file in files:
                        self.stats['bytes_saved'] += self.get_file_size(file)
                    self.stats['temporary_files'] += len(files)
                else:
                    try:
                        shutil.rmtree(demo_dir)
                        self.stats['temporary_files'] += len(files)
                        for file in files:
                            self.stats['bytes_saved'] += self.get_file_size(file)
                    except Exception as e:
                        print(f"   ❌ Error removing {demo_dir}: {e}")
                        return False
        
        return True
    
    def display_summary(self, analysis: Dict[str, List[Path]]):
        """Display analysis summary."""
        print("\n" + "="*60)
        print("📊 RESULTS DIRECTORY ANALYSIS")
        print("="*60)
        
        # Essential files
        print(f"\n✅ ESSENTIAL FILES ({len(analysis['essential'])} files):")
        for file in sorted(analysis['essential']):
            size = self.get_file_size(file)
            rel_path = file.relative_to(RESULTS_DIR)
            print(f"   📄 {rel_path} ({size:,} bytes)")
            self.stats['essential_files'] += 1
        
        # Temporary files
        print(f"\n🗑️  TEMPORARY FILES ({len(analysis['temporary'])} items):")
        temp_size = 0
        for file in sorted(analysis['temporary']):
            if file.is_file():
                size = self.get_file_size(file)
                temp_size += size
                rel_path = file.relative_to(RESULTS_DIR)
                print(f"   📄 {rel_path} ({size:,} bytes)")
            elif file.is_dir():
                size = sum(f.stat().st_size for f in file.rglob('*') if f.is_file())
                temp_size += size
                rel_path = file.relative_to(RESULTS_DIR)
                print(f"   📁 {rel_path}/ ({size:,} bytes)")
        
        # Demo files  
        if analysis.get('demo'):
            print(f"\n🎬 DEMO FILES ({len(analysis['demo'])} files):")
            demo_size = 0
            for file in sorted(analysis['demo']):
                size = self.get_file_size(file)
                demo_size += size
                rel_path = file.relative_to(RESULTS_DIR)
                print(f"   📄 {rel_path} ({size:,} bytes)")
            temp_size += demo_size

        # Unknown files (only show if any exist)
        unknown_non_gitkeep = [f for f in analysis['unknown'] if f.name != '.gitkeep']
        if unknown_non_gitkeep:
            print(f"\n❓ UNKNOWN FILES ({len(unknown_non_gitkeep)} files):")
            for file in sorted(unknown_non_gitkeep):
                size = self.get_file_size(file)
                rel_path = file.relative_to(RESULTS_DIR)
                print(f"   ❓ {rel_path} ({size:,} bytes)")
        
        print(f"\n💾 Total space to be freed: {temp_size:,} bytes ({temp_size/1024/1024:.1f} MB)")
        
        if self.dry_run:
            print("\n🔍 This is a DRY RUN - no files will be modified")
    
    def display_final_stats(self):
        """Display final cleanup statistics."""
        print("\n" + "="*60)
        print("📈 CLEANUP STATISTICS")
        print("="*60)
        print(f"Essential files identified: {self.stats['essential_files']}")
        print(f"Files backed up: {self.stats['backed_up']}")
        print(f"Temporary items removed: {self.stats['temporary_files']}")
        print(f"Space freed: {self.stats['bytes_saved']:,} bytes ({self.stats['bytes_saved']/1024/1024:.1f} MB)")
        
        if not self.dry_run:
            print(f"\n✅ Results directory cleaned and organized!")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Clean and organize the results directory for capstone presentation",
        epilog="""
Examples:
  %(prog)s --dry-run                    # Preview what will happen
  %(prog)s --backup                     # Create presentation backup only
  %(prog)s --clean                      # Remove temporary files only  
  %(prog)s --full                       # Backup + clean (recommended)
  %(prog)s --clean --keep-latest-demo   # Clean but keep latest demo folder
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview changes without modifying files")
    parser.add_argument("--backup", action="store_true",
                       help="Create presentation backup only")
    parser.add_argument("--clean", action="store_true", 
                       help="Remove temporary files only")
    parser.add_argument("--full", action="store_true",
                       help="Create backup AND clean temporary files")
    parser.add_argument("--keep-latest-demo", action="store_true",
                       help="Keep only the latest demo folder (when cleaning)")
    
    args = parser.parse_args()
    
    # Validate arguments
    action_count = sum([args.backup, args.clean, args.full])
    if action_count == 0 and not args.dry_run:
        parser.error("Must specify at least one action: --backup, --clean, --full, or --dry-run")
    if action_count > 1:
        parser.error("Can only specify one action at a time")
    
    # Initialize cleaner
    cleaner = ResultsCleaner(dry_run=args.dry_run)
    
    # Analyze directory
    print("🔍 Analyzing results directory...")
    analysis = cleaner.analyze_directory()
    
    if not any(analysis.values()):
        print("❌ No files found in results directory")
        return 1
    
    # Display analysis
    cleaner.display_summary(analysis)
    
    # Perform requested actions
    if args.dry_run:
        print("\n🔍 DRY RUN COMPLETE - No files were modified")
        return 0
    
    if args.backup or args.full:
        success = cleaner.create_backup(analysis['essential'])
        if not success:
            return 1
    
    if args.clean or args.full:
        success = cleaner.clean_temporary(analysis['temporary'])
        if not success:
            return 1
        
        # Handle demo files
        if analysis.get('demo'):
            success = cleaner.clean_demo_files(analysis['demo'], keep_latest=args.keep_latest_demo)
            if not success:
                return 1
    
    # Display final statistics
    cleaner.display_final_stats()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())