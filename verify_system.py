#!/usr/bin/env python3
"""
Mental Health LLM Evaluation System Verification
===============================================

This script performs a comprehensive verification of the entire system
to ensure all components are working correctly after the path consistency fix.
"""

import sys
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
import traceback
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class SystemVerifier:
    """Comprehensive system verification class."""
    
    def __init__(self):
        self.project_root = project_root
        self.results = {
            'imports': [],
            'directories': [],
            'paths': [],
            'config': [],
            'results': [],
            'warnings': [],
            'errors': []
        }
        self.success_count = 0
        self.total_tests = 0
    
    def log_success(self, category: str, message: str):
        """Log a successful test."""
        self.results[category].append(('✅', message))
        self.success_count += 1
        self.total_tests += 1
    
    def log_warning(self, category: str, message: str):
        """Log a warning."""
        self.results[category].append(('⚠️', message))
        self.results['warnings'].append(message)
        self.total_tests += 1
    
    def log_error(self, category: str, message: str):
        """Log an error."""
        self.results[category].append(('❌', message))
        self.results['errors'].append(message)
        self.total_tests += 1
    
    def print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{'='*70}")
        print(f"🔍 {title}")
        print(f"{'='*70}")
    
    def print_section(self, title: str):
        """Print a formatted section header."""
        print(f"\n{'-'*50}")
        print(f"📋 {title}")
        print(f"{'-'*50}")
    
    def test_imports(self):
        """Test all critical imports."""
        self.print_section("Testing System Imports")
        
        # Test paths.py import
        try:
            sys.path.insert(0, str(self.project_root / "src" / "utils"))
            import paths
            self.log_success('imports', "paths.py imported successfully")
        except Exception as e:
            self.log_error('imports', f"Failed to import paths.py: {e}")
        
        # Test MentalHealthEvaluator import
        try:
            from src.evaluation.mental_health_evaluator import MentalHealthEvaluator
            self.log_success('imports', "MentalHealthEvaluator imported successfully")
        except Exception as e:
            self.log_error('imports', f"Failed to import MentalHealthEvaluator: {e}")
        
        # Test ModelComparator import
        try:
            from scripts.compare_models import ModelComparator
            self.log_success('imports', "ModelComparator imported successfully")
        except Exception as e:
            self.log_error('imports', f"Failed to import ModelComparator: {e}")
        
        # Test statistical analysis import
        try:
            from src.analysis.statistical_analysis import analyze_results
            self.log_success('imports', "Statistical analysis imported successfully")
        except Exception as e:
            self.log_error('imports', f"Failed to import statistical analysis: {e}")
        
        # Test visualization import
        try:
            from src.analysis.visualization import create_all_visualizations
            self.log_success('imports', "Visualization tools imported successfully")
        except Exception as e:
            self.log_error('imports', f"Failed to import visualization tools: {e}")
    
    def test_directories(self):
        """Test all expected directories exist."""
        self.print_section("Verifying Directory Structure")
        
        # Get paths instance
        try:
            import paths
            paths_obj = paths.ProjectPaths()
        except Exception as e:
            self.log_error('directories', f"Failed to initialize paths: {e}")
            return
        
        # Test base directories
        base_dirs = {
            'Project Root': paths_obj.get_project_root(),
            'Source Directory': paths_obj.get_src_dir(),
            'Data Directory': paths_obj.get_data_dir(),
            'Results Directory': paths_obj.get_results_dir(),
            'Temp Directory': paths_obj.get_temp_dir(),
            'Config Directory': paths_obj.get_config_dir(),
            'Scripts Directory': paths_obj.get_scripts_dir(),
            'Docs Directory': paths_obj.get_docs_dir()
        }
        
        for name, path in base_dirs.items():
            if path.exists():
                self.log_success('directories', f"{name}: {path}")
            else:
                self.log_error('directories', f"{name} missing: {path}")
        
        # Test results subdirectories
        results_dirs = {
            'Evaluations': paths_obj.get_evaluations_dir(),
            'Reports': paths_obj.get_reports_dir(),
            'Statistics': paths_obj.get_statistics_dir(),
            'Visualizations': paths_obj.get_visualizations_dir(),
            'Conversations': paths_obj.get_conversations_dir(),
            'Development': paths_obj.get_development_dir()
        }
        
        for name, path in results_dirs.items():
            if path.exists():
                self.log_success('directories', f"Results/{name}: {path}")
            else:
                self.log_error('directories', f"Results/{name} missing: {path}")
        
        # No more generated subdirectories - everything moved to results/
    
    def test_paths_module(self):
        """Test paths module functionality."""
        self.print_section("Testing Paths Module")
        
        try:
            import paths
            paths_obj = paths.ProjectPaths()
            
            # Test convenience functions
            convenience_funcs = [
                'get_results_dir', 'get_evaluations_dir', 'get_reports_dir',
                'get_statistics_dir', 'get_visualizations_dir', 'get_conversations_dir', 
                'get_development_dir', 'get_temp_dir'
            ]
            
            for func_name in convenience_funcs:
                try:
                    func = getattr(paths, func_name)
                    result = func()
                    self.log_success('paths', f"{func_name}() returns: {result}")
                except Exception as e:
                    self.log_error('paths', f"{func_name}() failed: {e}")
            
            # Test file path generators
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                eval_file = paths_obj.get_evaluation_results_file(timestamp)
                report_file = paths_obj.get_analysis_report_file(timestamp)
                stats_file = paths_obj.get_statistical_analysis_file(timestamp)
                
                self.log_success('paths', f"File path generation working")
            except Exception as e:
                self.log_error('paths', f"File path generation failed: {e}")
            
            # Test utility methods
            try:
                test_dir = paths_obj.get_temp_dir() / "system_test"
                paths_obj.ensure_dir(test_dir)
                if test_dir.exists():
                    test_dir.rmdir()
                    self.log_success('paths', "Directory creation utility working")
                else:
                    self.log_error('paths', "Directory creation utility failed")
            except Exception as e:
                self.log_error('paths', f"Directory creation test failed: {e}")
                
        except Exception as e:
            self.log_error('paths', f"Paths module test failed: {e}")
    
    def test_configuration(self):
        """Test configuration files accessibility."""
        self.print_section("Testing Configuration Files")
        
        # Test main configuration
        main_config = self.project_root / "config" / "main.yaml"
        if main_config.exists():
            try:
                with open(main_config, 'r') as f:
                    config = yaml.safe_load(f)
                self.log_success('config', f"Main config loaded: {len(config)} sections")
                
                # Check for updated paths in config
                if 'paths' in config:
                    paths_config = config['paths']
                    if 'results_directory' in paths_config:
                        self.log_success('config', "Config uses updated results_directory")
                    else:
                        self.log_warning('config', "Config still uses old output_directory")
                    
                    if 'temp_directory' in paths_config:
                        self.log_success('config', "Config uses updated temp_directory")
                    else:
                        self.log_warning('config', "Config missing temp_directory")
                    
                    if 'development_directory' in paths_config:
                        self.log_success('config', "Config uses updated development_directory")
                    else:
                        self.log_warning('config', "Config missing development_directory")
                
            except Exception as e:
                self.log_error('config', f"Failed to load main config: {e}")
        else:
            self.log_error('config', "Main config file missing")
        
        # Test scenarios config
        scenarios_config = self.project_root / "config" / "scenarios" / "main_scenarios.yaml"
        if scenarios_config.exists():
            try:
                with open(scenarios_config, 'r') as f:
                    scenarios = yaml.safe_load(f)
                scenario_count = len(scenarios.get('scenarios', []))
                self.log_success('config', f"Scenarios config loaded: {scenario_count} scenarios")
            except Exception as e:
                self.log_error('config', f"Failed to load scenarios config: {e}")
        else:
            self.log_error('config', "Scenarios config file missing")
        
        # Test models config
        models_config = self.project_root / "config" / "models" / "model_settings.yaml"
        if models_config.exists():
            try:
                with open(models_config, 'r') as f:
                    models = yaml.safe_load(f)
                self.log_success('config', "Models config loaded successfully")
            except Exception as e:
                self.log_error('config', f"Failed to load models config: {e}")
        else:
            self.log_warning('config', "Models config file missing (optional)")
    
    def test_recent_results(self):
        """Test that recent results can be loaded."""
        self.print_section("Testing Recent Results")
        
        results_dir = self.project_root / "results"
        
        # Check for recent evaluation results
        eval_dir = results_dir / "evaluations"
        if eval_dir.exists():
            eval_files = list(eval_dir.glob("*.json"))
            if eval_files:
                try:
                    # Try to load the most recent file
                    recent_file = max(eval_files, key=lambda f: f.stat().st_mtime)
                    with open(recent_file, 'r') as f:
                        data = json.load(f)
                    self.log_success('results', f"Recent evaluation results loaded: {recent_file.name}")
                except Exception as e:
                    self.log_error('results', f"Failed to load evaluation results: {e}")
            else:
                self.log_warning('results', "No evaluation result files found")
        else:
            self.log_error('results', "Evaluations directory missing")
        
        # Check for recent reports
        reports_dir = results_dir / "reports"
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.txt"))
            if report_files:
                try:
                    recent_file = max(report_files, key=lambda f: f.stat().st_mtime)
                    content = recent_file.read_text()
                    self.log_success('results', f"Recent report loaded: {recent_file.name}")
                except Exception as e:
                    self.log_error('results', f"Failed to load report: {e}")
            else:
                self.log_warning('results', "No report files found")
        else:
            self.log_error('results', "Reports directory missing")
        
        # Check for recent statistics
        stats_dir = results_dir / "statistics"
        if stats_dir.exists():
            stats_files = list(stats_dir.glob("*.json"))
            if stats_files:
                try:
                    recent_file = max(stats_files, key=lambda f: f.stat().st_mtime)
                    with open(recent_file, 'r') as f:
                        data = json.load(f)
                    self.log_success('results', f"Recent statistics loaded: {recent_file.name}")
                except Exception as e:
                    self.log_error('results', f"Failed to load statistics: {e}")
            else:
                self.log_warning('results', "No statistics files found")
        else:
            self.log_error('results', "Statistics directory missing")
        
        # Check for visualizations
        viz_dir = results_dir / "visualizations"
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.jpg")) + list(viz_dir.glob("*.svg"))
            if viz_files:
                self.log_success('results', f"Visualizations found: {len(viz_files)} files")
            else:
                self.log_warning('results', "No visualization files found")
        else:
            self.log_error('results', "Visualizations directory missing")
    
    def test_minimal_evaluation(self):
        """Test that the evaluation system can be initialized."""
        self.print_section("Testing Evaluation System")
        
        try:
            from src.evaluation.mental_health_evaluator import MentalHealthEvaluator
            
            # Test initialization
            evaluator = MentalHealthEvaluator(models=['openai', 'deepseek'])
            self.log_success('results', "MentalHealthEvaluator initialized successfully")
            
            # Test scenario loading
            if len(evaluator.scenarios) > 0:
                self.log_success('results', f"Scenarios loaded: {len(evaluator.scenarios)} scenarios")
            else:
                self.log_warning('results', "No scenarios loaded")
            
        except Exception as e:
            self.log_error('results', f"Evaluation system test failed: {e}")
        
        try:
            from scripts.compare_models import ModelComparator
            
            # Test ModelComparator initialization
            comparator = ModelComparator(selected_models=['openai', 'deepseek'], quiet=True)
            self.log_success('results', "ModelComparator initialized successfully")
            
        except Exception as e:
            self.log_error('results', f"ModelComparator test failed: {e}")
    
    def run_verification(self):
        """Run all verification tests."""
        self.print_header("MENTAL HEALTH LLM EVALUATION SYSTEM VERIFICATION")
        
        # Run all tests
        self.test_imports()
        self.test_directories()
        self.test_paths_module()
        self.test_configuration()
        self.test_recent_results()
        self.test_minimal_evaluation()
        
        # Print summary report
        self.print_summary()
    
    def print_summary(self):
        """Print the final verification summary."""
        self.print_header("SYSTEM VERIFICATION REPORT")
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Successful: {self.success_count}")
        print(f"   Warnings: {len(self.results['warnings'])}")
        print(f"   Errors: {len(self.results['errors'])}")
        print(f"   Success Rate: {(self.success_count/self.total_tests)*100:.1f}%")
        
        # Print detailed results
        for category, items in self.results.items():
            if category in ['warnings', 'errors'] or not items:
                continue
            
            print(f"\n{category.upper()}:")
            for status, message in items:
                print(f"   {status} {message}")
        
        # Print warnings if any
        if self.results['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.results['warnings']:
                print(f"   • {warning}")
        
        # Print errors if any
        if self.results['errors']:
            print(f"\n❌ ERRORS:")
            for error in self.results['errors']:
                print(f"   • {error}")
        
        # Final status
        print(f"\n{'='*70}")
        if len(self.results['errors']) == 0:
            if len(self.results['warnings']) == 0:
                print("🎉 SYSTEM VERIFICATION: COMPLETE SUCCESS!")
                print("✅ All systems operational and ready for use")
            else:
                print("✅ SYSTEM VERIFICATION: SUCCESS WITH WARNINGS")
                print("⚠️  System is functional but some optimizations possible")
        else:
            print("❌ SYSTEM VERIFICATION: ISSUES FOUND")
            print("🔧 Please address the errors above before using the system")
        
        print(f"{'='*70}")
        
        return len(self.results['errors']) == 0

def main():
    """Main function."""
    verifier = SystemVerifier()
    success = verifier.run_verification()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())