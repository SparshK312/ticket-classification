#!/usr/bin/env python3
"""
Targeted debugging to capture exact error details
"""

import pandas as pd
from gretel_client.navigator_client import Gretel
import gretel_client.data_designer.params as P
import gretel_client.data_designer.columns as C

def test_llm_step_specifically():
    """Test specifically the LLM text generation step that's failing"""
    
    print("🎯 TARGETED LLM DEBUGGING")
    print("="*50)
    
    # Initialize
    gretel = Gretel(api_key="grtu10f9fc7866d4727a7d3108f7c61b115a33bdb0604805c2f33eec1c2d93130f4b")
    
    # Simple test data
    test_data = pd.DataFrame({
        'description': ['Password reset needed'],
        'category': ['User Account']
    })
    
    print("1. Setting up minimal workflow...")
    
    # Test with different model suites
    model_suites = ["apache-2.0", None]  # None = default
    
    for suite in model_suites:
        suite_name = suite if suite else "default"
        print(f"\n🧪 Testing with {suite_name} model suite...")
        
        try:
            # Create designer
            if suite:
                aidd = gretel.data_designer.new(model_suite=suite)
            else:
                aidd = gretel.data_designer.new()
            
            # Add seed data
            aidd.with_seed_dataset(test_data, sampling_strategy="shuffle")
            
            # Add non-LLM column first
            aidd.add_column(
                C.SamplerColumn(
                    name="variation",
                    type=P.SamplerType.CATEGORY,
                    params=P.CategorySamplerParams(
                        values=["v1", "v2"],
                        weights=[50, 50]
                    )
                )
            )
            
            # NOW try the LLM column - this is where it should fail
            print(f"   Adding LLM column to {suite_name}...")
            aidd.add_column(
                name="synthetic_desc",
                prompt="Rewrite: {{ description }}"
            )
            
            print(f"   ✅ LLM column added to {suite_name}")
            
            # Validate
            print(f"   Validating {suite_name}...")
            aidd.validate()
            print(f"   ✅ Validation passed for {suite_name}")
            
            # Create workflow - this should trigger the error
            print(f"   Creating workflow with {suite_name}...")
            workflow = aidd.create(
                num_records=1,
                name=f"debug-{suite_name}-llm-test"
            )
            
            print(f"   ✅ Workflow created with {suite_name}")
            print(f"   Console: {workflow.console_url}")
            
            # Wait and capture the exact error
            print(f"   Waiting for {suite_name} workflow to complete...")
            try:
                workflow.wait_until_done()
                result = workflow.dataset.df
                print(f"   🎉 SUCCESS with {suite_name}! Generated {len(result)} records")
                
                # Show result
                if len(result) > 0:
                    row = result.iloc[0]
                    print(f"   Original: {row.get('description', 'N/A')}")
                    print(f"   Synthetic: {row.get('synthetic_desc', 'N/A')}")
                
                return True  # Success!
                
            except Exception as workflow_error:
                print(f"   ❌ {suite_name} workflow failed: {workflow_error}")
                print(f"   Error type: {type(workflow_error).__name__}")
                
                # Try to get more details
                if hasattr(workflow_error, '__dict__'):
                    print(f"   Error attributes: {workflow_error.__dict__}")
                
                # Continue to test next model suite
                
        except Exception as setup_error:
            print(f"   ❌ {suite_name} setup failed: {setup_error}")
            print(f"   Setup error type: {type(setup_error).__name__}")
    
    return False

def test_alternative_approaches():
    """Test if other Gretel features work"""
    
    print("\n🔄 TESTING ALTERNATIVE APPROACHES")
    print("="*50)
    
    gretel = Gretel(api_key="grtu10f9fc7866d4727a7d3108f7c61b115a33bdb0604805c2f33eec1c2d93130f4b")
    
    # Test 1: Basic tabular generation (no LLM)
    print("1. Testing basic tabular generation...")
    try:
        test_data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B'],
            'priority': [1, 2, 1, 3],
            'status': ['open', 'closed', 'open', 'open']
        })
        
        aidd = gretel.data_designer.new()
        aidd.with_seed_dataset(test_data)
        
        # Only use samplers and expressions, no LLM
        aidd.add_column(
            C.SamplerColumn(
                name="new_priority",
                type=P.SamplerType.NUMERICAL,
                params=P.NumericalSamplerParams(min=1, max=5)
            )
        )
        
        aidd.validate()
        workflow = aidd.create(num_records=2, name="debug-tabular-only")
        workflow.wait_until_done()
        
        result = workflow.dataset.df
        print(f"   ✅ Basic tabular generation works: {len(result)} records")
        return True
        
    except Exception as e:
        print(f"   ❌ Basic tabular generation failed: {e}")
        return False

def main():
    """Run targeted debugging"""
    
    # Test the specific LLM step
    llm_success = test_llm_step_specifically()
    
    # Test alternative approaches
    alt_success = test_alternative_approaches()
    
    print(f"\n📊 FINAL RESULTS")
    print("="*50)
    print(f"LLM text generation: {'✅ WORKS' if llm_success else '❌ FAILS'}")
    print(f"Basic tabular generation: {'✅ WORKS' if alt_success else '❌ FAILS'}")
    
    if not llm_success and alt_success:
        print("\n🎯 DIAGNOSIS: LLM text generation service is down/broken")
        print("   Basic Gretel functionality works fine")
        print("   This is definitely a service-side issue with text generation")
        
    elif not llm_success and not alt_success:
        print("\n🎯 DIAGNOSIS: Broader Gretel service issues")
        print("   Multiple Gretel services appear to be affected")
        
    elif llm_success:
        print("\n🎯 DIAGNOSIS: LLM text generation is working!")
        print("   The issue might have been resolved or was temporary")
        
    print("\n📝 For support ticket, include:")
    print("   - Error: UnknownError(1203) in generate_column_from_template_v2")
    print("   - Scope: LLM text generation specifically") 
    print("   - Impact: Cannot generate synthetic text columns")
    print("   - Reproducible: Yes, consistently fails at same step")

if __name__ == "__main__":
    main()