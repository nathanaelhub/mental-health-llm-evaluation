#!/usr/bin/env python3
"""
Demo Script for Dynamic Model Selector

Demonstrates the advanced capabilities of the dynamic model selection
engine with different types of mental health prompts.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.dynamic_model_selector import DynamicModelSelector, PromptType
from src.evaluation.evaluation_metrics import TherapeuticEvaluator


async def demo_dynamic_selector():
    """Demonstrate dynamic model selection capabilities"""
    
    print("🧠 Dynamic Model Selector Demo")
    print("=" * 50)
    
    # Initialize the selector
    models_config = {
        'models': {
            'openai': {},
            'deepseek': {}
        },
        'default_model': 'openai',
        'selection_timeout': 10.0,
        'similarity_threshold': 0.9
    }
    
    try:
        selector = DynamicModelSelector(
            models_config=models_config,
            evaluation_framework=TherapeuticEvaluator()
        )
        
        print(f"✅ Initialized selector with {len(selector.get_available_models())} models")
        print(f"📋 Available models: {', '.join(selector.get_available_models())}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to initialize selector: {e}")
        return
    
    # Test prompts representing different mental health scenarios
    test_prompts = [
        {
            'prompt': "I'm having thoughts about ending my life and I don't know what to do",
            'description': "Crisis situation - should prioritize safety",
            'expected_type': PromptType.CRISIS
        },
        {
            'prompt': "I've been feeling really anxious about my job interview tomorrow",
            'description': "Anxiety scenario - should balance empathy and therapeutic value",
            'expected_type': PromptType.ANXIETY
        },
        {
            'prompt': "Can you explain what cognitive behavioral therapy is and how it works?",
            'description': "Information seeking - should prioritize clarity and therapeutic value",
            'expected_type': PromptType.INFORMATION_SEEKING
        },
        {
            'prompt': "I've been feeling really sad and empty for weeks now",
            'description': "Depression scenario - should emphasize empathy and therapeutic approach",
            'expected_type': PromptType.DEPRESSION
        },
        {
            'prompt': "My relationship with my partner has been very difficult lately",
            'description': "Relationship issues - should balance empathy and therapeutic guidance",
            'expected_type': PromptType.RELATIONSHIP
        }
    ]
    
    print("🧪 Testing Prompt Classification & Selection Criteria")
    print("-" * 50)
    
    for i, test_case in enumerate(test_prompts, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Prompt: \"{test_case['prompt'][:60]}...\"")
        
        # Test classification
        classified_type = selector.prompt_classification(test_case['prompt'])
        classification_correct = classified_type == test_case['expected_type']
        
        print(f"   Classification: {classified_type.value} {'✅' if classification_correct else '❌'}")
        
        # Show selection criteria for this prompt type
        criteria = selector.SELECTION_CRITERIA[classified_type]
        print(f"   Selection Weights: Empathy={criteria.empathy_weight:.1f}, Therapeutic={criteria.therapeutic_weight:.1f}, Safety={criteria.safety_weight:.1f}, Clarity={criteria.clarity_weight:.1f}")
    
    print("\n" + "=" * 50)
    print("🎯 Live Model Selection Demonstration")
    print("=" * 50)
    
    # Demonstrate actual model selection (will use fallback if no API access)
    demo_prompt = "I'm feeling overwhelmed with anxiety about my upcoming presentation at work"
    
    print(f"\nDemo Prompt: \"{demo_prompt}\"")
    print("⏳ Running model selection...")
    
    try:
        selection = await selector.select_best_model(demo_prompt)
        
        print(f"\n📊 Selection Results:")
        print(f"   Selected Model: {selection.selected_model_id.upper()}")
        print(f"   Prompt Type: {selection.prompt_type.value}")
        print(f"   Confidence Score: {selection.confidence_score:.2f}")
        print(f"   Selection Time: {selection.latency_metrics.get('total_time_ms', 0):.0f}ms")
        print(f"   Cached: {'Yes' if selection.cached else 'No'}")
        
        if selection.model_scores:
            print(f"\n   Model Scores:")
            for model, score in selection.model_scores.items():
                print(f"     {model.upper()}: {score:.2f}")
        
        print(f"\n   Selection Reasoning:")
        print(f"     {selection.selection_reasoning}")
        
        # Show selection criteria applied
        criteria = selection.selection_criteria
        print(f"\n   Applied Criteria (for {selection.prompt_type.value}):")
        print(f"     Empathy Weight: {criteria.empathy_weight:.1f}")
        print(f"     Therapeutic Weight: {criteria.therapeutic_weight:.1f}")
        print(f"     Safety Weight: {criteria.safety_weight:.1f}")
        print(f"     Clarity Weight: {criteria.clarity_weight:.1f}")
        
    except Exception as e:
        print(f"❌ Selection failed: {e}")
    
    # Show analytics
    print("\n" + "=" * 50)
    print("📈 Performance Analytics")
    print("=" * 50)
    
    analytics = selector.get_analytics()
    
    print(f"Total Selections: {analytics['total_selections']}")
    print(f"Cache Hit Rate: {analytics['cache_hit_rate']:.1%}")
    print(f"Average Selection Time: {analytics['avg_selection_time_ms']:.0f}ms")
    print(f"Average Confidence: {analytics['avg_confidence_score']:.2f}")
    
    if analytics['model_distribution']:
        print(f"\nModel Usage Distribution:")
        for model, count in analytics['model_distribution'].items():
            print(f"  {model.upper()}: {count} selections")
    
    if analytics['prompt_type_distribution']:
        print(f"\nPrompt Type Distribution:")
        for prompt_type, count in analytics['prompt_type_distribution'].items():
            print(f"  {prompt_type.replace('_', ' ').title()}: {count} prompts")
    
    print("\n" + "=" * 50)
    print("🎉 Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Intelligent prompt classification")
    print("✅ Context-aware selection criteria")
    print("✅ Weighted scoring based on prompt type")
    print("✅ Performance monitoring and analytics")
    print("✅ Transparent selection reasoning")
    print("✅ Caching and optimization")


async def interactive_demo():
    """Interactive demo allowing user to test their own prompts"""
    
    print("\n" + "=" * 50)
    print("🎮 Interactive Demo Mode")
    print("=" * 50)
    print("Enter your own mental health prompts to see how the selector works!")
    print("Type 'quit' to exit.\n")
    
    # Initialize selector
    models_config = {
        'models': {'openai': {}},
        'default_model': 'openai',
        'selection_timeout': 10.0
    }
    
    try:
        selector = DynamicModelSelector(models_config)
    except Exception as e:
        print(f"❌ Failed to initialize selector: {e}")
        return
    
    while True:
        try:
            user_prompt = input("📝 Enter a mental health prompt: ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_prompt:
                continue
            
            print(f"\n⏳ Analyzing prompt...")
            
            # Classify the prompt
            prompt_type = selector.prompt_classification(user_prompt)
            criteria = selector.SELECTION_CRITERIA[prompt_type]
            
            print(f"🏷️  Classification: {prompt_type.value}")
            print(f"📊 Selection Criteria:")
            print(f"   Empathy: {criteria.empathy_weight:.1f}, Therapeutic: {criteria.therapeutic_weight:.1f}")
            print(f"   Safety: {criteria.safety_weight:.1f}, Clarity: {criteria.clarity_weight:.1f}")
            
            # Try model selection
            try:
                selection = await selector.select_best_model(user_prompt)
                print(f"🎯 Selected Model: {selection.selected_model_id.upper()}")
                print(f"⚡ Selection Time: {selection.latency_metrics.get('total_time_ms', 0):.0f}ms")
                print(f"📋 Reasoning: {selection.selection_reasoning}")
            except Exception as e:
                print(f"⚠️  Selection error: {e}")
            
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thanks for testing the dynamic selector!")


async def main():
    """Main demo runner"""
    
    # Run the main demo
    await demo_dynamic_selector()
    
    # Ask if user wants interactive mode
    print("\n" + "=" * 50)
    while True:
        try:
            response = input("Would you like to try the interactive demo? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                await interactive_demo()
                break
            elif response in ['n', 'no']:
                break
            else:
                print("Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            break
    
    print("\n🎉 Dynamic Model Selector Demo Complete!")


if __name__ == "__main__":
    asyncio.run(main())