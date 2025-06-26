#!/usr/bin/env python3
"""
Generate Report Script

Creates comprehensive research report with results compilation, recommendations,
executive summary, and export in multiple formats (HTML, PDF, Markdown).

Usage:
    python scripts/generate_report.py --experiment exp_20240101_12345678
    python scripts/generate_report.py --experiment exp_20240101_12345678 --format pdf
    python scripts/generate_report.py --experiment exp_20240101_12345678 --template academic
    python scripts/generate_report.py --dry-run --experiment exp_20240101_12345678
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template
import markdown
import weasyprint

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.logging_config import setup_logging, get_logger


class ReportGenerator:
    """Generates comprehensive research reports."""
    
    def __init__(self, experiment_id: str, dry_run: bool = False):
        self.experiment_id = experiment_id
        self.dry_run = dry_run
        self.logger = get_logger(__name__)
        
        # Initialize state
        self.experiment_dir = None
        self.manifest = None
        self.evaluation_data = None
        self.analysis_results = None
        self.visualization_files = {}
        
        # Report data
        self.report_data = {}
        self.templates_dir = PROJECT_ROOT / "templates" / "reports"
        
    def load_experiment_data(self) -> bool:
        """Load all experiment data and results."""
        try:
            # Find experiment directory
            experiments_dir = PROJECT_ROOT / "experiments"
            self.experiment_dir = experiments_dir / self.experiment_id
            
            if not self.experiment_dir.exists():
                # Try finding by partial ID
                matching_dirs = [d for d in experiments_dir.iterdir() 
                               if d.is_dir() and self.experiment_id in d.name]
                if len(matching_dirs) == 1:
                    self.experiment_dir = matching_dirs[0]
                    self.experiment_id = matching_dirs[0].name
                elif len(matching_dirs) > 1:
                    self.logger.error(f"Multiple experiments match '{self.experiment_id}':")
                    for d in matching_dirs:
                        self.logger.error(f"  - {d.name}")
                    return False
                else:
                    self.logger.error(f"Experiment not found: {self.experiment_id}")
                    return False
            
            # Load manifest
            manifest_path = self.experiment_dir / "experiment_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    self.manifest = json.load(f)
                self.logger.info("Loaded experiment manifest")
            
            # Load evaluation results
            eval_path = self.experiment_dir / "evaluations" / "evaluation_results.json"
            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    self.evaluation_data = json.load(f)
                self.logger.info(f"Loaded {len(self.evaluation_data)} evaluation results")
            
            # Load analysis results
            analysis_path = self.experiment_dir / "results" / "statistical_analysis.json"
            if analysis_path.exists():
                with open(analysis_path, 'r') as f:
                    self.analysis_results = json.load(f)
                self.logger.info("Loaded statistical analysis results")
            
            # Load model comparison summary
            model_summary_path = self.experiment_dir / "results" / "model_comparison_summary.json"
            if model_summary_path.exists():
                with open(model_summary_path, 'r') as f:
                    self.model_summary = json.load(f)
                self.logger.info("Loaded model comparison summary")
            
            # Find visualization files
            viz_dir = self.experiment_dir / "results" / "visualizations"
            if viz_dir.exists():
                for viz_file in viz_dir.iterdir():
                    if viz_file.is_file():
                        self.visualization_files[viz_file.stem] = viz_file
                self.logger.info(f"Found {len(self.visualization_files)} visualization files")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load experiment data: {str(e)}")
            return False
    
    def prepare_report_data(self) -> bool:
        """Prepare all data for report generation."""
        try:
            self.report_data = {
                "metadata": self._prepare_metadata(),
                "executive_summary": self._prepare_executive_summary(),
                "methodology": self._prepare_methodology(),
                "results": self._prepare_results(),
                "statistical_analysis": self._prepare_statistical_analysis(),
                "model_comparison": self._prepare_model_comparison(),
                "safety_analysis": self._prepare_safety_analysis(),
                "recommendations": self._prepare_recommendations(),
                "limitations": self._prepare_limitations(),
                "conclusions": self._prepare_conclusions(),
                "appendices": self._prepare_appendices()
            }
            
            self.logger.info("Prepared report data")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare report data: {str(e)}")
            return False
    
    def _prepare_metadata(self) -> Dict[str, Any]:
        """Prepare report metadata."""
        return {
            "title": "Mental Health LLM Evaluation Report",
            "experiment_id": self.experiment_id,
            "generated_at": datetime.now().isoformat(),
            "generated_by": "Mental Health LLM Evaluation Framework",
            "version": "1.0.0",
            "experiment_config": self.manifest.get("configuration", {}) if self.manifest else {},
            "experiment_created": self.manifest.get("created_at", "") if self.manifest else ""
        }
    
    def _prepare_executive_summary(self) -> Dict[str, Any]:
        """Prepare executive summary."""
        summary = {
            "overview": "",
            "key_findings": [],
            "recommendations": [],
            "implications": []
        }
        
        try:
            if self.evaluation_data:
                successful_evals = [e for e in self.evaluation_data if e.get("status") == "completed"]
                total_conversations = len(successful_evals)
                
                models = list(set(e.get("model_name", "") for e in successful_evals))
                scenarios = list(set(e.get("scenario_id", "") for e in successful_evals))
                
                summary["overview"] = (
                    f"This report presents a comprehensive evaluation of {len(models)} "
                    f"Large Language Models across {len(scenarios)} mental health scenarios, "
                    f"analyzing {total_conversations} conversations. The evaluation assessed "
                    f"technical performance, therapeutic quality, and safety considerations."
                )
                
                # Extract key findings from model summary
                if hasattr(self, 'model_summary') and self.model_summary:
                    best_overall = None
                    best_safety = None
                    
                    for model, data in self.model_summary.items():
                        if isinstance(data, dict) and "performance_metrics" in data:
                            overall_score = data["performance_metrics"].get("overall_score", {}).get("mean", 0)
                            safety_score = data["performance_metrics"].get("safety_score", {}).get("mean", 0)
                            
                            if not best_overall or overall_score > best_overall[1]:
                                best_overall = (model, overall_score)
                            if not best_safety or safety_score > best_safety[1]:
                                best_safety = (model, safety_score)
                    
                    if best_overall:
                        summary["key_findings"].append(
                            f"{best_overall[0]} achieved the highest overall quality score ({best_overall[1]:.2f}/10)"
                        )
                    
                    if best_safety:
                        summary["key_findings"].append(
                            f"{best_safety[0]} demonstrated the best safety performance ({best_safety[1]:.2f}/10)"
                        )
                    
                    # Add comparative analysis if available
                    if "comparative_analysis" in self.model_summary:
                        comp_analysis = self.model_summary["comparative_analysis"]
                        if "recommendation" in comp_analysis:
                            summary["recommendations"].append(comp_analysis["recommendation"])
                
                summary["implications"] = [
                    "Local models may offer privacy advantages but require careful performance evaluation",
                    "Cloud-based models typically provide higher quality but raise data privacy concerns",
                    "Safety monitoring is critical for all models in mental health applications",
                    "Regular validation against clinical standards is essential for deployment"
                ]
        
        except Exception as e:
            self.logger.warning(f"Failed to prepare executive summary: {str(e)}")
        
        return summary
    
    def _prepare_methodology(self) -> Dict[str, Any]:
        """Prepare methodology section."""
        methodology = {
            "overview": "",
            "models_evaluated": [],
            "evaluation_metrics": [],
            "scenarios": [],
            "statistical_methods": []
        }
        
        try:
            if self.manifest:
                config = self.manifest.get("configuration", {})
                
                # Models
                models_config = config.get("models", {})
                for model_name, model_config in models_config.items():
                    if model_config.get("enabled", False):
                        methodology["models_evaluated"].append({
                            "name": model_name,
                            "type": "Cloud-based API" if model_name == "openai" else "Local inference",
                            "configuration": model_config
                        })
                
                # Scenarios
                if "scenarios" in self.manifest:
                    methodology["scenarios"] = [
                        {
                            "id": s["scenario_id"],
                            "title": s["title"],
                            "category": s["category"],
                            "severity": s["severity"]
                        } for s in self.manifest["scenarios"]
                    ]
                
                # Evaluation configuration
                eval_config = config.get("evaluation", {})
                methodology["evaluation_metrics"] = [
                    "Empathy scoring (0-10 scale)",
                    "Safety detection and crisis assessment",
                    "Coherence and consistency evaluation",
                    "Therapeutic technique assessment",
                    "Composite quality scoring"
                ]
                
                methodology["statistical_methods"] = [
                    "Analysis of Variance (ANOVA) for model comparison",
                    "Pairwise t-tests with Bonferroni correction",
                    "Cohen's d effect size calculation",
                    "Correlation analysis",
                    "Descriptive statistics and confidence intervals"
                ]
                
                methodology["overview"] = (
                    f"The evaluation methodology employed a comprehensive framework "
                    f"assessing {len(methodology['models_evaluated'])} models across "
                    f"{len(methodology['scenarios'])} standardized mental health scenarios. "
                    f"Each conversation was evaluated using multiple validated metrics "
                    f"with statistical analysis to ensure reliable comparisons."
                )
        
        except Exception as e:
            self.logger.warning(f"Failed to prepare methodology: {str(e)}")
        
        return methodology
    
    def _prepare_results(self) -> Dict[str, Any]:
        """Prepare results section."""
        results = {
            "overview": "",
            "performance_summary": {},
            "detailed_metrics": {},
            "visualizations": []
        }
        
        try:
            if self.evaluation_data:
                successful_evals = [e for e in self.evaluation_data if e.get("status") == "completed"]
                
                # Performance summary
                models = {}
                for eval_result in successful_evals:
                    model_name = eval_result.get("model_name", "unknown")
                    if model_name not in models:
                        models[model_name] = {
                            "conversation_count": 0,
                            "scores": {
                                "empathy": [],
                                "safety": [],
                                "coherence": [],
                                "overall": []
                            }
                        }
                    
                    models[model_name]["conversation_count"] += 1
                    
                    # Extract scores
                    scores = eval_result.get("scores", {})
                    if "empathy" in scores:
                        empathy_score = scores["empathy"].get("average_score", 0) if isinstance(scores["empathy"], dict) else scores["empathy"]
                        models[model_name]["scores"]["empathy"].append(empathy_score)
                    
                    if "safety" in scores:
                        safety_score = scores["safety"].get("safety_score", 0) if isinstance(scores["safety"], dict) else scores["safety"]
                        models[model_name]["scores"]["safety"].append(safety_score)
                    
                    if "coherence" in scores:
                        coherence_score = scores["coherence"].get("average_score", 0) if isinstance(scores["coherence"], dict) else scores["coherence"]
                        models[model_name]["scores"]["coherence"].append(coherence_score)
                    
                    if "composite" in scores:
                        overall_score = scores["composite"].get("overall_score", 0) if isinstance(scores["composite"], dict) else scores["composite"]
                        models[model_name]["scores"]["overall"].append(overall_score)
                
                # Calculate summary statistics
                for model_name, data in models.items():
                    summary = {}
                    for metric, scores in data["scores"].items():
                        if scores:
                            summary[metric] = {
                                "mean": sum(scores) / len(scores),
                                "std": (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5,
                                "min": min(scores),
                                "max": max(scores),
                                "count": len(scores)
                            }
                    
                    results["performance_summary"][model_name] = {
                        "conversations": data["conversation_count"],
                        "metrics": summary
                    }
                
                results["overview"] = (
                    f"A total of {len(successful_evals)} conversations were successfully "
                    f"evaluated across {len(models)} models. The results demonstrate "
                    f"varying performance across different evaluation dimensions."
                )
                
                # Add visualization references
                results["visualizations"] = [
                    {
                        "title": "Model Comparison Box Plots",
                        "file": "model_comparison_boxplots.png",
                        "description": "Distribution of scores across models for key metrics"
                    },
                    {
                        "title": "Correlation Heatmap",
                        "file": "correlation_heatmap.png",
                        "description": "Correlation matrix showing relationships between metrics"
                    },
                    {
                        "title": "Safety Analysis",
                        "file": "safety_analysis.png",
                        "description": "Safety score distributions and flag analysis by model"
                    },
                    {
                        "title": "Performance vs Quality",
                        "file": "performance_vs_quality.png",
                        "description": "Scatter plot showing trade-offs between response time and quality"
                    }
                ]
        
        except Exception as e:
            self.logger.warning(f"Failed to prepare results: {str(e)}")
        
        return results
    
    def _prepare_statistical_analysis(self) -> Dict[str, Any]:
        """Prepare statistical analysis section."""
        analysis = {
            "overview": "",
            "anova_results": {},
            "pairwise_comparisons": [],
            "effect_sizes": {},
            "significance_summary": []
        }
        
        try:
            if self.analysis_results:
                analysis["anova_results"] = self.analysis_results.get("anova_results", {})
                
                # Extract pairwise comparisons
                pairwise_data = self.analysis_results.get("pairwise_comparisons", {})
                if pairwise_data:
                    analysis["pairwise_comparisons"] = [
                        {
                            "comparison": key,
                            "p_value": value.get("p_value", 1.0),
                            "effect_size": value.get("effect_size", 0.0),
                            "significant": value.get("significant", False),
                            "effect_magnitude": value.get("effect_magnitude", "negligible")
                        }
                        for key, value in pairwise_data.items()
                        if isinstance(value, dict)
                    ]
                
                # Summarize significant findings
                significant_comparisons = [
                    comp for comp in analysis["pairwise_comparisons"]
                    if comp["significant"]
                ]
                
                if significant_comparisons:
                    analysis["significance_summary"] = [
                        f"{comp['comparison']}: p={comp['p_value']:.3f}, "
                        f"effect size={comp['effect_size']:.2f} ({comp['effect_magnitude']})"
                        for comp in significant_comparisons
                    ]
                
                analysis["overview"] = (
                    f"Statistical analysis revealed {len(significant_comparisons)} "
                    f"significant differences out of {len(analysis['pairwise_comparisons'])} "
                    f"pairwise comparisons performed."
                )
        
        except Exception as e:
            self.logger.warning(f"Failed to prepare statistical analysis: {str(e)}")
        
        return analysis
    
    def _prepare_model_comparison(self) -> Dict[str, Any]:
        """Prepare detailed model comparison."""
        comparison = {
            "overview": "",
            "model_rankings": {},
            "strengths_weaknesses": {},
            "use_case_recommendations": {}
        }
        
        try:
            if hasattr(self, 'model_summary') and self.model_summary:
                models_data = {}
                
                for model, data in self.model_summary.items():
                    if isinstance(data, dict) and "performance_metrics" in data:
                        models_data[model] = data
                
                # Rank models by different criteria
                criteria = ["overall_score", "empathy_score", "safety_score", "coherence_score"]
                
                for criterion in criteria:
                    rankings = []
                    for model, data in models_data.items():
                        score = data["performance_metrics"].get(criterion, {}).get("mean", 0)
                        rankings.append((model, score))
                    
                    rankings.sort(key=lambda x: x[1], reverse=True)
                    comparison["model_rankings"][criterion] = rankings
                
                # Analyze strengths and weaknesses
                for model, data in models_data.items():
                    strengths = []
                    weaknesses = []
                    
                    metrics = data["performance_metrics"]
                    safety_data = data.get("safety_analysis", {})
                    
                    # Identify strengths (scores > 7.5)
                    if metrics.get("overall_score", {}).get("mean", 0) > 7.5:
                        strengths.append("High overall quality")
                    if metrics.get("empathy_score", {}).get("mean", 0) > 7.5:
                        strengths.append("Strong empathy recognition")
                    if metrics.get("safety_score", {}).get("mean", 0) > 8.0:
                        strengths.append("Excellent safety performance")
                    if metrics.get("response_time_ms", {}).get("mean", 5000) < 2000:
                        strengths.append("Fast response times")
                    
                    # Identify weaknesses
                    if safety_data.get("safety_concern_rate", 0) > 10:
                        weaknesses.append("Higher rate of safety concerns")
                    if metrics.get("coherence_score", {}).get("mean", 0) < 6.0:
                        weaknesses.append("Lower coherence scores")
                    if metrics.get("response_time_ms", {}).get("mean", 0) > 4000:
                        weaknesses.append("Slower response times")
                    
                    comparison["strengths_weaknesses"][model] = {
                        "strengths": strengths,
                        "weaknesses": weaknesses
                    }
                
                # Use case recommendations
                if "comparative_analysis" in self.model_summary:
                    comp_analysis = self.model_summary["comparative_analysis"]
                    
                    comparison["use_case_recommendations"] = {
                        "clinical_applications": comp_analysis.get("best_safety", ""),
                        "research_applications": comp_analysis.get("best_overall_quality", ""),
                        "high_volume_applications": comp_analysis.get("fastest_response", ""),
                        "general_recommendation": comp_analysis.get("recommendation", "")
                    }
                
                comparison["overview"] = (
                    f"Detailed comparison analysis of {len(models_data)} models "
                    f"reveals distinct performance profiles and optimal use cases "
                    f"for different application scenarios."
                )
        
        except Exception as e:
            self.logger.warning(f"Failed to prepare model comparison: {str(e)}")
        
        return comparison
    
    def _prepare_safety_analysis(self) -> Dict[str, Any]:
        """Prepare safety analysis section."""
        safety = {
            "overview": "",
            "safety_metrics": {},
            "crisis_detection": {},
            "safety_recommendations": []
        }
        
        try:
            if self.evaluation_data:
                safety_data = {}
                crisis_data = {}
                
                for eval_result in self.evaluation_data:
                    if eval_result.get("status") != "completed":
                        continue
                    
                    model_name = eval_result.get("model_name", "unknown")
                    
                    if model_name not in safety_data:
                        safety_data[model_name] = {
                            "total_conversations": 0,
                            "safety_flags": 0,
                            "crisis_flags": 0,
                            "low_safety_scores": 0
                        }
                    
                    safety_data[model_name]["total_conversations"] += 1
                    
                    # Extract safety information
                    scores = eval_result.get("scores", {})
                    flags = eval_result.get("flags", [])
                    
                    safety_data[model_name]["safety_flags"] += len(flags)
                    
                    # Count crisis flags
                    crisis_flags = [f for f in flags if f.get("level") == "crisis"]
                    safety_data[model_name]["crisis_flags"] += len(crisis_flags)
                    
                    # Check for low safety scores
                    safety_score = 10  # Default
                    if "safety" in scores:
                        safety_score = scores["safety"].get("safety_score", 10) if isinstance(scores["safety"], dict) else scores["safety"]
                    
                    if safety_score < 7:
                        safety_data[model_name]["low_safety_scores"] += 1
                
                # Calculate safety metrics
                for model, data in safety_data.items():
                    total = data["total_conversations"]
                    if total > 0:
                        safety["safety_metrics"][model] = {
                            "conversations": total,
                            "safety_flag_rate": data["safety_flags"] / total,
                            "crisis_detection_rate": data["crisis_flags"] / total,
                            "low_safety_score_rate": data["low_safety_scores"] / total * 100
                        }
                
                safety["overview"] = (
                    "Safety analysis is critical for mental health applications. "
                    "This section examines crisis detection capabilities, safety flag "
                    "rates, and overall safety performance across models."
                )
                
                safety["safety_recommendations"] = [
                    "Implement human oversight for all conversations with safety flags",
                    "Establish clear escalation protocols for crisis situations",
                    "Regularly validate crisis detection accuracy against expert assessments",
                    "Maintain comprehensive audit logs for safety-related incidents",
                    "Provide immediate access to crisis resources and hotlines"
                ]
        
        except Exception as e:
            self.logger.warning(f"Failed to prepare safety analysis: {str(e)}")
        
        return safety
    
    def _prepare_recommendations(self) -> Dict[str, Any]:
        """Prepare recommendations section."""
        recommendations = {
            "deployment_recommendations": [],
            "technical_recommendations": [],
            "clinical_recommendations": [],
            "research_recommendations": []
        }
        
        try:
            # Deployment recommendations
            recommendations["deployment_recommendations"] = [
                "Implement comprehensive safety monitoring for all deployed models",
                "Establish clear protocols for human intervention in crisis situations",
                "Regular performance monitoring and model revalidation",
                "Compliance with healthcare regulations (HIPAA, GDPR)",
                "User consent and transparency about AI assistance"
            ]
            
            # Technical recommendations
            recommendations["technical_recommendations"] = [
                "Optimize model performance for real-time response requirements",
                "Implement robust error handling and fallback mechanisms",
                "Regular security audits and vulnerability assessments",
                "Scalable infrastructure for varying usage patterns",
                "Comprehensive logging and monitoring systems"
            ]
            
            # Clinical recommendations
            recommendations["clinical_recommendations"] = [
                "Licensed mental health professional oversight required",
                "Regular validation against clinical standards",
                "Clear scope of practice limitations",
                "Integration with existing clinical workflows",
                "Ongoing training for healthcare providers using AI tools"
            ]
            
            # Research recommendations
            recommendations["research_recommendations"] = [
                "Longitudinal studies of patient outcomes",
                "Cross-cultural validation of evaluation metrics",
                "Development of standardized evaluation benchmarks",
                "Investigation of bias in AI mental health applications",
                "Collaboration with mental health professionals for validation"
            ]
            
        except Exception as e:
            self.logger.warning(f"Failed to prepare recommendations: {str(e)}")
        
        return recommendations
    
    def _prepare_limitations(self) -> Dict[str, Any]:
        """Prepare limitations section."""
        return {
            "study_limitations": [
                "Evaluation based on synthetic scenarios rather than real patient data",
                "Limited to English language conversations",
                "Focus on text-based interactions only",
                "Evaluation metrics may not capture all aspects of therapeutic quality",
                "Limited longitudinal assessment of therapeutic outcomes"
            ],
            "technical_limitations": [
                "Model performance may vary with different hardware configurations",
                "API-based models subject to service availability and rate limits",
                "Local models require significant computational resources",
                "Evaluation framework requires ongoing validation and updates"
            ],
            "generalizability_limitations": [
                "Results may not generalize to all mental health conditions",
                "Cultural and linguistic diversity not fully represented",
                "Professional practice variations not fully captured",
                "Regulatory requirements vary by jurisdiction"
            ]
        }
    
    def _prepare_conclusions(self) -> Dict[str, Any]:
        """Prepare conclusions section."""
        conclusions = {
            "key_findings": [],
            "implications": [],
            "future_work": []
        }
        
        try:
            # Extract key findings from analysis
            if hasattr(self, 'model_summary') and self.model_summary:
                if "comparative_analysis" in self.model_summary:
                    comp_analysis = self.model_summary["comparative_analysis"]
                    
                    conclusions["key_findings"] = [
                        f"Model performance varies significantly across evaluation dimensions",
                        f"Safety considerations are paramount for mental health applications",
                        f"Trade-offs exist between performance, quality, and privacy",
                        f"Both local and cloud-based models show promise for specific use cases"
                    ]
                    
                    if "recommendation" in comp_analysis:
                        conclusions["key_findings"].append(comp_analysis["recommendation"])
            
            conclusions["implications"] = [
                "AI systems can augment but not replace human mental health professionals",
                "Comprehensive evaluation frameworks are essential for safe deployment",
                "Ongoing monitoring and validation are critical for maintaining quality",
                "Stakeholder involvement is crucial for successful implementation"
            ]
            
            conclusions["future_work"] = [
                "Development of real-world validation studies",
                "Integration with clinical decision support systems",
                "Expansion to multilingual and multicultural contexts",
                "Investigation of personalization and adaptation techniques",
                "Long-term outcome studies and effectiveness research"
            ]
            
        except Exception as e:
            self.logger.warning(f"Failed to prepare conclusions: {str(e)}")
        
        return conclusions
    
    def _prepare_appendices(self) -> Dict[str, Any]:
        """Prepare appendices section."""
        appendices = {
            "statistical_tables": {},
            "configuration_details": {},
            "visualization_index": [],
            "technical_specifications": {}
        }
        
        try:
            # Configuration details
            if self.manifest:
                appendices["configuration_details"] = self.manifest.get("configuration", {})
            
            # Visualization index
            for viz_name, viz_path in self.visualization_files.items():
                appendices["visualization_index"].append({
                    "name": viz_name,
                    "filename": viz_path.name,
                    "description": f"Visualization: {viz_name.replace('_', ' ').title()}"
                })
            
            # Technical specifications
            appendices["technical_specifications"] = {
                "framework_version": "1.0.0",
                "evaluation_metrics": [
                    "Empathy Scoring", "Safety Detection", "Coherence Evaluation",
                    "Therapeutic Assessment", "Composite Scoring"
                ],
                "statistical_methods": [
                    "ANOVA", "t-tests", "Effect Size Analysis", "Correlation Analysis"
                ]
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to prepare appendices: {str(e)}")
        
        return appendices
    
    def create_html_report(self, template_name: str = "default") -> Path:
        """Create HTML report."""
        try:
            # Create HTML template if not exists
            html_template = self._get_html_template(template_name)
            
            # Render template
            html_content = html_template.render(**self.report_data)
            
            # Save HTML report
            reports_dir = self.experiment_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            html_path = reports_dir / f"evaluation_report_{template_name}.html"
            
            if not self.dry_run:
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.info(f"Created HTML report: {html_path}")
            else:
                self.logger.info(f"DRY RUN: Would create HTML report at {html_path}")
            
            return html_path
            
        except Exception as e:
            self.logger.error(f"Failed to create HTML report: {str(e)}")
            raise
    
    def create_pdf_report(self, html_path: Path) -> Path:
        """Create PDF report from HTML."""
        try:
            pdf_path = html_path.with_suffix('.pdf')
            
            if not self.dry_run:
                # Convert HTML to PDF using weasyprint
                html_doc = weasyprint.HTML(filename=str(html_path))
                html_doc.write_pdf(str(pdf_path))
                self.logger.info(f"Created PDF report: {pdf_path}")
            else:
                self.logger.info(f"DRY RUN: Would create PDF report at {pdf_path}")
            
            return pdf_path
            
        except Exception as e:
            self.logger.error(f"Failed to create PDF report: {str(e)}")
            raise
    
    def create_markdown_report(self) -> Path:
        """Create Markdown report."""
        try:
            markdown_template = self._get_markdown_template()
            markdown_content = markdown_template.render(**self.report_data)
            
            reports_dir = self.experiment_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            md_path = reports_dir / "evaluation_report.md"
            
            if not self.dry_run:
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                self.logger.info(f"Created Markdown report: {md_path}")
            else:
                self.logger.info(f"DRY RUN: Would create Markdown report at {md_path}")
            
            return md_path
            
        except Exception as e:
            self.logger.error(f"Failed to create Markdown report: {str(e)}")
            raise
    
    def _get_html_template(self, template_name: str) -> Template:
        """Get HTML template."""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ metadata.title }}</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1, h2, h3 { color: #2c3e50; }
        h1 { border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }
        .metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .summary { background: #e8f5e8; padding: 15px; border-radius: 5px; border-left: 4px solid #27ae60; }
        .findings { background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; }
        .recommendations { background: #d1ecf1; padding: 15px; border-radius: 5px; border-left: 4px solid #17a2b8; }
        .limitations { background: #f8d7da; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric-score { font-weight: bold; }
        .score-high { color: #27ae60; }
        .score-medium { color: #f39c12; }
        .score-low { color: #e74c3c; }
        ul, ol { padding-left: 20px; }
        .visualization { text-align: center; margin: 20px 0; }
        .visualization img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="metadata">
        <h1>{{ metadata.title }}</h1>
        <p><strong>Experiment ID:</strong> {{ metadata.experiment_id }}</p>
        <p><strong>Generated:</strong> {{ metadata.generated_at }}</p>
        <p><strong>Framework Version:</strong> {{ metadata.version }}</p>
    </div>

    <div class="summary">
        <h2>Executive Summary</h2>
        <p>{{ executive_summary.overview }}</p>
        
        <h3>Key Findings</h3>
        <ul>
        {% for finding in executive_summary.key_findings %}
            <li>{{ finding }}</li>
        {% endfor %}
        </ul>
        
        <h3>Primary Recommendations</h3>
        <ul>
        {% for rec in executive_summary.recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
    </div>

    <h2>Methodology</h2>
    <p>{{ methodology.overview }}</p>
    
    <h3>Models Evaluated</h3>
    <table>
        <tr><th>Model</th><th>Type</th><th>Configuration</th></tr>
        {% for model in methodology.models_evaluated %}
        <tr>
            <td>{{ model.name }}</td>
            <td>{{ model.type }}</td>
            <td>{{ model.configuration.model if model.configuration.model else 'Default' }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Results Overview</h2>
    <p>{{ results.overview }}</p>
    
    <h3>Performance Summary</h3>
    <table>
        <tr>
            <th>Model</th>
            <th>Conversations</th>
            <th>Overall Score</th>
            <th>Empathy Score</th>
            <th>Safety Score</th>
            <th>Coherence Score</th>
        </tr>
        {% for model, data in results.performance_summary.items() %}
        <tr>
            <td>{{ model }}</td>
            <td>{{ data.conversations }}</td>
            <td class="metric-score {% if data.metrics.overall.mean >= 8 %}score-high{% elif data.metrics.overall.mean >= 6 %}score-medium{% else %}score-low{% endif %}">
                {{ "%.2f"|format(data.metrics.overall.mean) }} ± {{ "%.2f"|format(data.metrics.overall.std) }}
            </td>
            <td class="metric-score">{{ "%.2f"|format(data.metrics.empathy.mean) }} ± {{ "%.2f"|format(data.metrics.empathy.std) }}</td>
            <td class="metric-score">{{ "%.2f"|format(data.metrics.safety.mean) }} ± {{ "%.2f"|format(data.metrics.safety.std) }}</td>
            <td class="metric-score">{{ "%.2f"|format(data.metrics.coherence.mean) }} ± {{ "%.2f"|format(data.metrics.coherence.std) }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Statistical Analysis</h2>
    <p>{{ statistical_analysis.overview }}</p>
    
    {% if statistical_analysis.significance_summary %}
    <h3>Significant Differences</h3>
    <ul>
    {% for significance in statistical_analysis.significance_summary %}
        <li>{{ significance }}</li>
    {% endfor %}
    </ul>
    {% endif %}

    <h2>Safety Analysis</h2>
    <div class="findings">
        <p>{{ safety_analysis.overview }}</p>
        
        <h3>Safety Metrics by Model</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>Conversations</th>
                <th>Safety Flag Rate</th>
                <th>Crisis Detection Rate</th>
                <th>Low Safety Score Rate (%)</th>
            </tr>
            {% for model, metrics in safety_analysis.safety_metrics.items() %}
            <tr>
                <td>{{ model }}</td>
                <td>{{ metrics.conversations }}</td>
                <td>{{ "%.3f"|format(metrics.safety_flag_rate) }}</td>
                <td>{{ "%.3f"|format(metrics.crisis_detection_rate) }}</td>
                <td>{{ "%.1f"|format(metrics.low_safety_score_rate) }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <h2>Model Comparison</h2>
    <p>{{ model_comparison.overview }}</p>
    
    <h3>Strengths and Weaknesses</h3>
    {% for model, analysis in model_comparison.strengths_weaknesses.items() %}
    <h4>{{ model }}</h4>
    <p><strong>Strengths:</strong></p>
    <ul>
    {% for strength in analysis.strengths %}
        <li>{{ strength }}</li>
    {% endfor %}
    </ul>
    <p><strong>Areas for Improvement:</strong></p>
    <ul>
    {% for weakness in analysis.weaknesses %}
        <li>{{ weakness }}</li>
    {% endfor %}
    </ul>
    {% endfor %}

    <div class="recommendations">
        <h2>Recommendations</h2>
        
        <h3>Deployment Recommendations</h3>
        <ul>
        {% for rec in recommendations.deployment_recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
        
        <h3>Clinical Recommendations</h3>
        <ul>
        {% for rec in recommendations.clinical_recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
        
        <h3>Technical Recommendations</h3>
        <ul>
        {% for rec in recommendations.technical_recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
    </div>

    <h2>Conclusions</h2>
    <h3>Key Findings</h3>
    <ul>
    {% for finding in conclusions.key_findings %}
        <li>{{ finding }}</li>
    {% endfor %}
    </ul>
    
    <h3>Implications</h3>
    <ul>
    {% for implication in conclusions.implications %}
        <li>{{ implication }}</li>
    {% endfor %}
    </ul>
    
    <h3>Future Work</h3>
    <ul>
    {% for work in conclusions.future_work %}
        <li>{{ work }}</li>
    {% endfor %}
    </ul>

    <div class="limitations">
        <h2>Limitations</h2>
        
        <h3>Study Limitations</h3>
        <ul>
        {% for limitation in limitations.study_limitations %}
            <li>{{ limitation }}</li>
        {% endfor %}
        </ul>
        
        <h3>Technical Limitations</h3>
        <ul>
        {% for limitation in limitations.technical_limitations %}
            <li>{{ limitation }}</li>
        {% endfor %}
        </ul>
        
        <h3>Generalizability Limitations</h3>
        <ul>
        {% for limitation in limitations.generalizability_limitations %}
            <li>{{ limitation }}</li>
        {% endfor %}
        </ul>
    </div>

    <h2>Appendices</h2>
    
    <h3>Configuration Details</h3>
    <pre><code>{{ appendices.configuration_details | tojson(indent=2) }}</code></pre>
    
    <h3>Visualizations Generated</h3>
    <ul>
    {% for viz in appendices.visualization_index %}
        <li><strong>{{ viz.name }}:</strong> {{ viz.description }}</li>
    {% endfor %}
    </ul>
    
    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #bdc3c7; text-align: center; color: #7f8c8d;">
        <p>Generated by Mental Health LLM Evaluation Framework v{{ metadata.version }}</p>
        <p>Report generated on {{ metadata.generated_at }}</p>
    </footer>
</body>
</html>
        """
        
        return Template(template_content)
    
    def _get_markdown_template(self) -> Template:
        """Get Markdown template."""
        template_content = """
# {{ metadata.title }}

**Experiment ID:** {{ metadata.experiment_id }}  
**Generated:** {{ metadata.generated_at }}  
**Framework Version:** {{ metadata.version }}

## Executive Summary

{{ executive_summary.overview }}

### Key Findings

{% for finding in executive_summary.key_findings %}
- {{ finding }}
{% endfor %}

### Primary Recommendations

{% for rec in executive_summary.recommendations %}
- {{ rec }}
{% endfor %}

## Methodology

{{ methodology.overview }}

### Models Evaluated

| Model | Type | Configuration |
|-------|------|---------------|
{% for model in methodology.models_evaluated %}
| {{ model.name }} | {{ model.type }} | {{ model.configuration.model if model.configuration.model else 'Default' }} |
{% endfor %}

### Evaluation Metrics

{% for metric in methodology.evaluation_metrics %}
- {{ metric }}
{% endfor %}

### Statistical Methods

{% for method in methodology.statistical_methods %}
- {{ method }}
{% endfor %}

## Results Overview

{{ results.overview }}

### Performance Summary

| Model | Conversations | Overall Score | Empathy Score | Safety Score | Coherence Score |
|-------|---------------|---------------|---------------|--------------|-----------------|
{% for model, data in results.performance_summary.items() %}
| {{ model }} | {{ data.conversations }} | {{ "%.2f"|format(data.metrics.overall.mean) }} ± {{ "%.2f"|format(data.metrics.overall.std) }} | {{ "%.2f"|format(data.metrics.empathy.mean) }} ± {{ "%.2f"|format(data.metrics.empathy.std) }} | {{ "%.2f"|format(data.metrics.safety.mean) }} ± {{ "%.2f"|format(data.metrics.safety.std) }} | {{ "%.2f"|format(data.metrics.coherence.mean) }} ± {{ "%.2f"|format(data.metrics.coherence.std) }} |
{% endfor %}

## Statistical Analysis

{{ statistical_analysis.overview }}

{% if statistical_analysis.significance_summary %}
### Significant Differences

{% for significance in statistical_analysis.significance_summary %}
- {{ significance }}
{% endfor %}
{% endif %}

## Safety Analysis

{{ safety_analysis.overview }}

### Safety Metrics by Model

| Model | Conversations | Safety Flag Rate | Crisis Detection Rate | Low Safety Score Rate (%) |
|-------|---------------|------------------|----------------------|---------------------------|
{% for model, metrics in safety_analysis.safety_metrics.items() %}
| {{ model }} | {{ metrics.conversations }} | {{ "%.3f"|format(metrics.safety_flag_rate) }} | {{ "%.3f"|format(metrics.crisis_detection_rate) }} | {{ "%.1f"|format(metrics.low_safety_score_rate) }}% |
{% endfor %}

### Safety Recommendations

{% for rec in safety_analysis.safety_recommendations %}
- {{ rec }}
{% endfor %}

## Model Comparison

{{ model_comparison.overview }}

### Strengths and Weaknesses

{% for model, analysis in model_comparison.strengths_weaknesses.items() %}
#### {{ model }}

**Strengths:**
{% for strength in analysis.strengths %}
- {{ strength }}
{% endfor %}

**Areas for Improvement:**
{% for weakness in analysis.weaknesses %}
- {{ weakness }}
{% endfor %}

{% endfor %}

## Recommendations

### Deployment Recommendations

{% for rec in recommendations.deployment_recommendations %}
- {{ rec }}
{% endfor %}

### Clinical Recommendations

{% for rec in recommendations.clinical_recommendations %}
- {{ rec }}
{% endfor %}

### Technical Recommendations

{% for rec in recommendations.technical_recommendations %}
- {{ rec }}
{% endfor %}

### Research Recommendations

{% for rec in recommendations.research_recommendations %}
- {{ rec }}
{% endfor %}

## Conclusions

### Key Findings

{% for finding in conclusions.key_findings %}
- {{ finding }}
{% endfor %}

### Implications

{% for implication in conclusions.implications %}
- {{ implication }}
{% endfor %}

### Future Work

{% for work in conclusions.future_work %}
- {{ work }}
{% endfor %}

## Limitations

### Study Limitations

{% for limitation in limitations.study_limitations %}
- {{ limitation }}
{% endfor %}

### Technical Limitations

{% for limitation in limitations.technical_limitations %}
- {{ limitation }}
{% endfor %}

### Generalizability Limitations

{% for limitation in limitations.generalizability_limitations %}
- {{ limitation }}
{% endfor %}

## Appendices

### Visualizations Generated

{% for viz in appendices.visualization_index %}
- **{{ viz.name }}:** {{ viz.description }}
{% endfor %}

---

*Generated by Mental Health LLM Evaluation Framework v{{ metadata.version }}*  
*Report generated on {{ metadata.generated_at }}*
        """
        
        return Template(template_content)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive report for Mental Health LLM Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate HTML report
  python scripts/generate_report.py --experiment exp_20240101_12345678
  
  # Generate PDF report
  python scripts/generate_report.py --experiment exp_20240101_12345678 --format pdf
  
  # Generate all formats
  python scripts/generate_report.py --experiment exp_20240101_12345678 --format all
  
  # Test run without generating files
  python scripts/generate_report.py --dry-run --experiment exp_20240101_12345678
        """
    )
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment ID to generate report for"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["html", "pdf", "markdown", "all"],
        default="html",
        help="Output format for the report"
    )
    
    parser.add_argument(
        "--template", "-t",
        type=str,
        choices=["default", "academic", "executive"],
        default="default",
        help="Report template to use"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run without generating actual report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    logger = get_logger(__name__)
    
    try:
        # Initialize report generator
        generator = ReportGenerator(
            experiment_id=args.experiment,
            dry_run=args.dry_run
        )
        
        # Load experiment data
        if not generator.load_experiment_data():
            return 1
        
        # Prepare report data
        if not generator.prepare_report_data():
            return 1
        
        # Display experiment info
        print(f"\n{'='*60}")
        print(f"Mental Health LLM Evaluation - Report Generation")
        print(f"{'='*60}")
        print(f"Experiment ID: {generator.experiment_id}")
        print(f"Report Format: {args.format}")
        print(f"Template: {args.template}")
        
        if args.dry_run:
            print("🧪 DRY RUN MODE - No reports will be generated")
        
        print()
        
        generated_files = []
        
        # Generate reports
        if args.format in ["html", "all"]:
            logger.info("Generating HTML report...")
            html_path = generator.create_html_report(args.template)
            generated_files.append(("HTML", html_path))
            
            # Generate PDF from HTML if requested
            if args.format in ["pdf", "all"]:
                logger.info("Generating PDF report...")
                try:
                    pdf_path = generator.create_pdf_report(html_path)
                    generated_files.append(("PDF", pdf_path))
                except Exception as e:
                    logger.warning(f"PDF generation failed: {e}")
                    print(f"⚠️  PDF generation failed: {e}")
        
        if args.format in ["markdown", "all"]:
            logger.info("Generating Markdown report...")
            md_path = generator.create_markdown_report()
            generated_files.append(("Markdown", md_path))
        
        print(f"\n✅ Report generation completed!")
        if not args.dry_run and generated_files:
            print(f"📊 Generated reports:")
            for format_name, file_path in generated_files:
                print(f"  - {format_name}: {file_path}")
            
            print(f"\n🎉 Evaluation pipeline completed successfully!")
            print(f"📁 All results available in: {generator.experiment_dir}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Report generation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())