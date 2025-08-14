#!/usr/bin/env python3
"""
Minimal test based on exactly what worked in the notebook
"""

import pandas as pd
from gretel_client.navigator_client import Gretel
import gretel_client.data_designer.params as P
import gretel_client.data_designer.columns as C

def main():
    print("🔬 MINIMAL GRETEL TEST - Exact Notebook Replication")
    print("="*50)
    
    # Initialize Gretel
    gretel = Gretel(api_key="grtu10f9fc7866d4727a7d3108f7c61b115a33bdb0604805c2f33eec1c2d93130f4b")
    
    # Create very simple test data - exactly like notebook
    test_data = pd.DataFrame({
        'short_description': [
            'Password reset required',
            'Account locked out',
            'Cannot access email'
        ],
        'category': ['User Account Management'] * 3,
        'child_category_1': ['Password Reset', 'Account Unlock', 'Login Issues']
    })
    
    print(f"Test data shape: {test_data.shape}")
    
    # Create Data Designer - exactly as in notebook
    aidd = gretel.data_designer.new(model_suite="apache-2.0")
    
    # Add seed dataset
    aidd.with_seed_dataset(
        test_data,
        sampling_strategy="shuffle", 
        with_replacement=True
    )
    
    # Simple variation sampler - exactly like notebook
    aidd.add_column(
        C.SamplerColumn(
            name="variation_type",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["different_wording", "similar_problem"],
                weights=[50, 50]
            )
        )
    )
    
    # Simple description generation - copy exact working prompt from notebook
    aidd.add_column(
        name="synthetic_description",
        prompt="""Based on this original ticket: {{ short_description }}

{% if variation_type == 'different_wording' %}
Rewrite this exact issue using different words but keep the same problem.
{% else %}
Create a different but related problem in the same category.
{% endif %}

Write naturally as an IT user would."""
    )
    
    # Validate
    print("Validating...")
    aidd.validate()
    print("✅ Validated")
    
    # Generate just 1 record
    print("Creating workflow...")
    workflow = aidd.create(num_records=1, name="minimal-test")
    
    print(f"Console URL: {workflow.console_url}")
    print("Waiting for completion...")
    
    try:
        workflow.wait_until_done()
        result = workflow.dataset.df
        print(f"✅ Success! Generated {len(result)} records")
        
        if len(result) > 0:
            print("\nResult:")
            print(f"Original: {result.iloc[0]['short_description']}")
            print(f"Synthetic: {result.iloc[0]['synthetic_description']}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()