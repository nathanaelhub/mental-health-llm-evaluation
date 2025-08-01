#!/usr/bin/env python3
"""
Mental Health AI Optimization System Demo

Comprehensive demonstration of the complete optimization system including:
- Smart semantic caching with FAISS
- Progressive enhancement with background evaluation
- Prompt classification shortcuts
- Model warm-up and optimization
- Batch processing for multiple users
- Performance monitoring and metrics
"""

import asyncio
import logging
import time
import random
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.dynamic_model_selector import DynamicModelSelector, PromptType
from src.optimization import (
    SmartModelCache, PerformanceMonitor, ProgressiveEnhancer,
    PromptShortcuts, WarmupManager, BatchProcessor,
    OptimizationConfig, BatchRequest, RequestPriority
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationSystemDemo:
    """Comprehensive demo of the optimization system"""
    
    def __init__(self):
        self.cache = None
        self.performance_monitor = None
        self.model_selector = None
        self.progressive_enhancer = None
        self.prompt_shortcuts = None
        self.warmup_manager = None
        self.batch_processor = None
        
        # Demo data
        self.demo_prompts = [
            # Crisis prompts
            ("I can't take this anymore, I want to end it all", PromptType.CRISIS),
            ("I'm thinking about hurting myself", PromptType.CRISIS),
            ("I feel like killing myself", PromptType.CRISIS),
            
            # Anxiety prompts
            ("I'm having a panic attack and can't breathe", PromptType.ANXIETY),
            ("I feel so anxious about everything", PromptType.ANXIETY),
            ("My heart is racing and I'm sweating", PromptType.ANXIETY),
            
            # Depression prompts
            ("I feel so empty and hopeless", PromptType.DEPRESSION),
            ("I can't get out of bed anymore", PromptType.DEPRESSION),
            ("Nothing brings me joy anymore", PromptType.DEPRESSION),
            
            # Information seeking
            ("What are the symptoms of depression?", PromptType.INFORMATION_SEEKING),
            ("How can I help someone with anxiety?", PromptType.INFORMATION_SEEKING),
            ("Explain cognitive behavioral therapy", PromptType.INFORMATION_SEEKING),
            
            # General wellness
            ("Hello, how are you today?", PromptType.GENERAL_WELLNESS),
            ("I want to practice mindfulness", PromptType.GENERAL_WELLNESS),
            ("What are some self-care tips?", PromptType.GENERAL_WELLNESS),
        ]
    
    async def initialize_system(self):
        """Initialize all optimization components"""
        logger.info("🚀 Initializing Mental Health AI Optimization System...")
        
        # Initialize core components
        self.cache = SmartModelCache(
            cache_dir="results/development/cache",
            similarity_threshold=0.85,
            max_cache_size=1000
        )
        
        self.performance_monitor = PerformanceMonitor(
            history_window_hours=24,
            alert_thresholds={
                'p99_latency_ms': 5000,
                'error_rate_percent': 5.0,
                'cache_hit_rate_min': 60.0
            }
        )
        
        # Mock model selector for demo
        self.model_selector = MockModelSelector()
        
        # Initialize optimization components
        self.progressive_enhancer = ProgressiveEnhancer(
            cache=self.cache,
            model_selector=self.model_selector,
            performance_monitor=self.performance_monitor,
            max_background_workers=2
        )
        
        self.prompt_shortcuts = PromptShortcuts()
        
        self.warmup_manager = WarmupManager(
            config=OptimizationConfig(
                warmup_strategy=WarmupStrategy.PREDICTIVE,
                max_loaded_models=5,
                enable_model_pooling=True
            )
        )
        
        self.batch_processor = BatchProcessor(
            model_selector=self.model_selector,
            cache=self.cache,
            performance_monitor=self.performance_monitor,
            config=BatchingConfig(
                strategy=BatchStrategy.ADAPTIVE,
                max_batch_size=10,
                max_concurrent_batches=3
            )
        )
        
        # Start background services
        await self.progressive_enhancer.start_background_workers()
        await self.warmup_manager.start()
        await self.batch_processor.start()
        self.performance_monitor.start_monitoring()
        
        logger.info("✅ All optimization components initialized successfully")
    
    async def demo_smart_caching(self):
        """Demonstrate smart semantic caching"""
        logger.info("\n📊 === SMART CACHING DEMO ===")
        
        # First request - cache miss
        prompt1 = "I feel really sad and hopeless"
        logger.info(f"🔍 Testing cache for: '{prompt1}'")
        
        cached_result = await self.cache.get_cached_selection(prompt1, PromptType.DEPRESSION)
        if cached_result:
            logger.info(f"✅ Cache HIT: {cached_result.selected_model}")
        else:
            logger.info("❌ Cache MISS - performing full evaluation")
            
            # Simulate model selection and store in cache
            mock_selection = MockModelSelection(
                selected_model="claude-3-sonnet",
                confidence_score=0.85,
                prompt_classification=PromptType.DEPRESSION,
                reasoning="High therapeutic understanding for depression"
            )
            
            await self.cache.store_selection(prompt1, mock_selection, success=True)
            logger.info("💾 Stored selection in cache")
        
        # Similar request - should hit cache
        prompt2 = "I'm feeling very sad and without hope"
        logger.info(f"🔍 Testing similar prompt: '{prompt2}'")
        
        cached_result = await self.cache.get_cached_selection(prompt2, PromptType.DEPRESSION)
        if cached_result:
            logger.info(f"✅ Cache HIT via semantic similarity: {cached_result.selected_model} (score: {cached_result.cache_hit_score:.2f})")
        else:
            logger.info("❌ Cache MISS - semantic similarity not sufficient")
        
        # Display cache statistics
        stats = self.cache.get_statistics()
        logger.info(f"📈 Cache Stats: {stats.hit_rate:.1f}% hit rate, {stats.total_requests} total requests")
    
    async def demo_prompt_shortcuts(self):
        """Demonstrate prompt classification shortcuts"""
        logger.info("\n⚡ === PROMPT SHORTCUTS DEMO ===")
        
        test_prompts = [
            "I want to kill myself",  # Should trigger crisis shortcut
            "I feel anxious about work",  # Should trigger anxiety shortcut
            "Hello, how are you?",  # Should trigger greeting shortcut
            "What is cognitive behavioral therapy?"  # Should trigger info shortcut
        ]
        
        for prompt in test_prompts:
            logger.info(f"🔍 Testing shortcut for: '{prompt}'")
            
            start_time = time.time()
            shortcut_result = self.prompt_shortcuts.classify_prompt(prompt)
            classification_time = (time.time() - start_time) * 1000
            
            if shortcut_result:
                logger.info(f"⚡ SHORTCUT: {shortcut_result.prompt_type.value} -> {shortcut_result.suggested_model}")
                logger.info(f"   Confidence: {shortcut_result.confidence:.2f}, Time: {classification_time:.2f}ms")
                logger.info(f"   Patterns: {shortcut_result.matched_patterns}")
            else:
                logger.info("❌ No shortcut found - would use full evaluation")
        
        # Display shortcut metrics
        metrics = self.prompt_shortcuts.get_metrics()
        logger.info(f"📊 Shortcut Stats: {metrics.hit_rate():.1f}% hit rate, {metrics.avg_classification_time_ms:.2f}ms avg time")
    
    async def demo_progressive_enhancement(self):
        """Demonstrate progressive enhancement system"""
        logger.info("\n🔄 === PROGRESSIVE ENHANCEMENT DEMO ===")
        
        prompt = "I'm feeling overwhelmed with stress and anxiety"
        logger.info(f"🔍 Progressive enhancement for: '{prompt}'")
        
        # Get fast selection
        start_time = time.time()
        fast_selection = await self.progressive_enhancer.select_model_enhanced(
            prompt=prompt,
            prompt_type=PromptType.ANXIETY
        )
        fast_time = (time.time() - start_time) * 1000
        
        logger.info(f"⚡ Fast selection: {fast_selection.selected_model} ({fast_selection.selection_method})")
        logger.info(f"   Confidence: {fast_selection.confidence_score:.2f}, Time: {fast_time:.2f}ms")
        
        if fast_selection.background_evaluation_id:
            logger.info(f"🔄 Background evaluation queued: {fast_selection.background_evaluation_id}")
            
            # Wait a bit for background evaluation
            await asyncio.sleep(2)
            
            # Check queue status
            queue_status = await self.progressive_enhancer.get_queue_status()
            logger.info(f"📋 Queue status: {queue_status}")
        
        # Display enhancement metrics
        metrics = self.progressive_enhancer.get_metrics()
        logger.info(f"📈 Enhancement Stats: {metrics.fast_vs_full_agreement_rate:.1f}% agreement rate")
    
    async def demo_model_warmup(self):
        """Demonstrate model warm-up system"""
        logger.info("\n🔥 === MODEL WARM-UP DEMO ===")
        
        # Test model warm-up
        models_to_warmup = ["claude-3-sonnet", "gpt-4-turbo", "claude-3-haiku"]
        
        for model in models_to_warmup:
            logger.info(f"🔥 Warming up model: {model}")
            success = await self.warmup_manager.warmup_model(model, priority=1, reason="demo")
            if success:
                logger.info(f"✅ {model} queued for warm-up")
            else:
                logger.info(f"❌ Failed to queue {model}")
        
        # Wait for warm-up to complete
        await asyncio.sleep(3)
        
        # Check model readiness
        for model in models_to_warmup:
            is_ready = await self.warmup_manager.ensure_model_ready(model, timeout=5.0)
            status = "✅ READY" if is_ready else "❌ NOT READY"
            logger.info(f"   {model}: {status}")
        
        # Display optimization insights
        insights = self.warmup_manager.get_optimization_insights()
        logger.info(f"🎯 Optimization Insights: {len(insights['recommendations'])} recommendations")
        for rec in insights['recommendations'][:3]:  # Show first 3
            logger.info(f"   • {rec['message']}")
    
    async def demo_batch_processing(self):
        """Demonstrate batch processing system"""
        logger.info("\n📦 === BATCH PROCESSING DEMO ===")
        
        # Create multiple user requests
        batch_requests = []
        users = ["user1", "user2", "user3", "user4", "user5"]
        
        for i, (prompt, prompt_type) in enumerate(self.demo_prompts[:10]):
            user_id = users[i % len(users)]
            
            # Vary priorities
            if "kill" in prompt.lower() or "hurt" in prompt.lower():
                priority = RequestPriority.CRITICAL
            elif prompt_type in [PromptType.ANXIETY, PromptType.DEPRESSION]:
                priority = RequestPriority.HIGH
            else:
                priority = RequestPriority.MEDIUM
            
            request_id = await self.batch_processor.submit_request(
                user_id=user_id,
                prompt=prompt,
                prompt_type=prompt_type,
                priority=priority,
                result_callback=self._batch_result_callback
            )
            
            batch_requests.append(request_id)
            logger.info(f"📝 Submitted request {request_id} for {user_id} (priority: {priority.name})")
        
        # Wait for processing
        logger.info("⏳ Waiting for batch processing...")
        await asyncio.sleep(5)
        
        # Display batch metrics
        metrics = self.batch_processor.get_metrics()
        logger.info(f"📊 Batch Stats:")
        logger.info(f"   Total requests: {metrics.total_requests}")
        logger.info(f"   Total batches: {metrics.total_batches}")
        logger.info(f"   Avg batch size: {metrics.avg_batch_size:.1f}")
        logger.info(f"   Requests/sec: {metrics.requests_per_second:.2f}")
        logger.info(f"   Cache hit rate: {metrics.cache_hit_rate:.1f}%")
        
        # Display queue status
        queue_status = self.batch_processor.get_queue_status()
        logger.info(f"📋 Queue Status: {queue_status['total_size']} requests, {queue_status['active_batches']} active batches")
    
    async def _batch_result_callback(self, result):
        """Callback for batch processing results"""
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        cache_info = "(cached)" if result.cache_hit else "(fresh)"
        logger.info(f"   📨 Result for {result.user_id}: {status} {cache_info} - {result.processing_time_ms:.1f}ms")
    
    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring"""
        logger.info("\n📈 === PERFORMANCE MONITORING DEMO ===")
        
        # Simulate some requests with different outcomes
        test_scenarios = [
            ("successful_request", True, 150.0, "claude-3-sonnet"),
            ("slow_request", True, 3500.0, "gpt-4-turbo"),
            ("failed_request", False, 1000.0, "claude-3-opus"),
            ("fast_cached_request", True, 25.0, "gpt-3.5-turbo"),
        ]
        
        for scenario_name, success, latency, model in test_scenarios:
            request_id = self.performance_monitor.start_request(scenario_name)
            
            # Simulate processing time
            await asyncio.sleep(latency / 1000.0)
            
            actual_latency = self.performance_monitor.end_request(
                request_id, 
                success=success, 
                model_used=model,
                operation_type="model_selection"
            )
            
            logger.info(f"📊 {scenario_name}: {actual_latency:.2f}ms {'✅' if success else '❌'} ({model})")
        
        # Get current metrics
        current_metrics = self.performance_monitor.get_current_metrics()
        logger.info(f"📈 Performance Metrics:")
        logger.info(f"   P95 Latency: {current_metrics['latency']['p95']:.2f}ms")
        logger.info(f"   Error Rate: {current_metrics['error_rate']:.1f}%")
        logger.info(f"   Throughput: {current_metrics['throughput']['requests_per_second']:.2f} RPS")
        
        # Check for alerts
        alerts = self.performance_monitor.check_alert_conditions()
        if alerts:
            logger.info(f"🚨 Performance Alerts:")
            for alert in alerts:
                logger.info(f"   {alert['type']}: {alert['message']}")
        else:
            logger.info("✅ No performance alerts")
        
        # Get cost optimization insights
        insights = self.performance_monitor.get_cost_optimization_insights()
        logger.info(f"💰 Cost Insights: {len(insights['recommendations'])} recommendations")
        for rec in insights['recommendations'][:3]:
            logger.info(f"   • {rec['description']}")
    
    async def demo_system_integration(self):
        """Demonstrate full system integration"""
        logger.info("\n🌟 === FULL SYSTEM INTEGRATION DEMO ===")
        
        # Test a complex user interaction flow
        user_session = [
            ("Hello, I need help", PromptType.GENERAL_WELLNESS),
            ("I've been feeling really anxious lately", PromptType.ANXIETY),
            ("What are some coping strategies for anxiety?", PromptType.INFORMATION_SEEKING),
            ("I sometimes feel like I can't go on", PromptType.DEPRESSION),  # Escalation
        ]
        
        logger.info("🎭 Simulating complex user session...")
        
        for i, (prompt, expected_type) in enumerate(user_session):
            logger.info(f"\n💬 User message {i+1}: '{prompt}'")
            
            # Step 1: Check shortcuts
            start_time = time.time()
            shortcut = self.prompt_shortcuts.classify_prompt(prompt)
            
            if shortcut and shortcut.bypass_full_evaluation:
                logger.info(f"⚡ Shortcut used: {shortcut.prompt_type.value} -> {shortcut.suggested_model}")
                processing_time = (time.time() - start_time) * 1000
            else:
                # Step 2: Try progressive enhancement
                fast_selection = await self.progressive_enhancer.select_model_enhanced(prompt, expected_type)
                logger.info(f"🔄 Progressive enhancement: {fast_selection.selected_model} ({fast_selection.selection_method})")
                processing_time = fast_selection.latency_ms
            
            # Record metrics
            request_id = f"session_req_{i}"
            self.performance_monitor.start_request(request_id)
            await asyncio.sleep(processing_time / 1000.0)  # Simulate processing
            self.performance_monitor.end_request(request_id, success=True, model_used="integrated")
            
            logger.info(f"⏱️  Total response time: {processing_time:.2f}ms")
        
        # Final system status
        logger.info("\n📊 Final System Status:")
        
        # Cache performance
        cache_stats = self.cache.get_statistics()
        logger.info(f"   Cache: {cache_stats.hit_rate:.1f}% hit rate")
        
        # Shortcut performance
        shortcut_metrics = self.prompt_shortcuts.get_metrics()
        logger.info(f"   Shortcuts: {shortcut_metrics.hit_rate():.1f}% hit rate")
        
        # Enhancement performance
        enhancement_metrics = self.progressive_enhancer.get_metrics()
        logger.info(f"   Enhancement: {enhancement_metrics.total_requests} requests processed")
        
        # Batch performance
        batch_metrics = self.batch_processor.get_metrics()
        logger.info(f"   Batching: {batch_metrics.avg_batch_size:.1f} avg batch size")
        
        # Overall performance
        perf_metrics = self.performance_monitor.get_current_metrics()
        logger.info(f"   Performance: {perf_metrics['latency']['mean']:.2f}ms avg latency")
    
    async def cleanup(self):
        """Clean up all system resources"""
        logger.info("\n🧹 Cleaning up system resources...")
        
        try:
            await self.progressive_enhancer.stop_background_workers()
            await self.warmup_manager.stop()
            await self.batch_processor.stop()
            await self.performance_monitor.stop_monitoring()
            await self.cache.cleanup()
            
            logger.info("✅ All resources cleaned up successfully")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    async def run_full_demo(self):
        """Run the complete optimization system demo"""
        try:
            await self.initialize_system()
            
            # Run individual component demos
            await self.demo_smart_caching()
            await self.demo_prompt_shortcuts()
            await self.demo_progressive_enhancement()
            await self.demo_model_warmup()
            await self.demo_batch_processing()
            await self.demo_performance_monitoring()
            
            # Show integrated system in action
            await self.demo_system_integration()
            
            logger.info("\n🎉 === DEMO COMPLETED SUCCESSFULLY ===")
            logger.info("The Mental Health AI Optimization System is fully functional!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            await self.cleanup()


# Mock classes for demo (replace with actual implementations)
class MockModelSelector:
    """Mock model selector for demo purposes"""
    
    async def select_model(self, prompt: str, prompt_type: PromptType = None):
        # Simulate model selection logic
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate processing time
        
        models = ["gpt-3.5-turbo", "gpt-4-turbo", "claude-3-sonnet", "claude-3-opus"]
        selected = random.choice(models)
        
        return MockModelSelection(
            selected_model=selected,
            confidence_score=random.uniform(0.7, 0.95),
            prompt_classification=prompt_type or PromptType.GENERAL_WELLNESS,
            reasoning=f"Mock selection: {selected} for {prompt_type.value if prompt_type else 'general'}"
        )


class MockModelSelection:
    """Mock model selection result"""
    
    def __init__(self, selected_model, confidence_score, prompt_classification, reasoning):
        self.selected_model = selected_model
        self.confidence_score = confidence_score
        self.prompt_classification = prompt_classification
        self.reasoning = reasoning
        self.evaluation_time_ms = random.uniform(100, 500)
        self.model_scores = {selected_model: confidence_score}


# Import missing enums for demo
from src.optimization.model_optimizer import WarmupStrategy
from src.optimization.batch_processor import BatchingConfig, BatchStrategy


async def main():
    """Main demo function"""
    print("🏥 Mental Health AI Optimization System Demo")
    print("=" * 60)
    
    demo = OptimizationSystemDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())