#!/usr/bin/env python3
"""
Debug LLM Response - See what Llama is actually outputting

Quick test to see the raw LLM response and fix JSON parsing issues.
"""

from llm_automation_classifier import LLMAutomationClassifier, AutomationFrameworkPrompts
import pandas as pd

def debug_single_response():
    """Test one problem group and show raw response"""
    
    # Load one problem group
    groups_df = pd.read_csv("outputs/improved_problem_groups.csv")
    sample_group = groups_df.iloc[0]  # First group for testing
    
    print(f"🔍 Debugging LLM response for:")
    print(f"PG_{sample_group['problem_group_id']}: {sample_group['representative_short_description']}")
    
    # Initialize classifier
    classifier = LLMAutomationClassifier()
    
    # Create the exact prompt that would be sent
    prompts = AutomationFrameworkPrompts()
    
    sample_descriptions = ["Sample ticket 1", "Sample ticket 2", "Sample ticket 3"]
    user_prompt = prompts.USER_PROMPT_TEMPLATE.format(
        problem_group_id=sample_group['problem_group_id'],
        representative_description=sample_group['representative_short_description'],
        ticket_count=sample_group['group_size'],
        quality_score=sample_group['quality_score'],
        sample_descriptions="\n".join([f"{i}. {desc}" for i, desc in enumerate(sample_descriptions, 1)])
    )
    
    print(f"\n📤 SENDING PROMPT:")
    print("=" * 50)
    print(f"SYSTEM: {prompts.SYSTEM_PROMPT[:200]}...")
    print("=" * 50)
    print(f"USER: {user_prompt[:500]}...")
    print("=" * 50)
    
    # Get raw response
    try:
        response = classifier.ollama_client.chat(
            model=classifier.model,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.1,
                "num_predict": 1000,
            }
        )
        
        raw_response = response['message']['content']
        
        print(f"\n📥 RAW LLM RESPONSE:")
        print("=" * 50)
        print(raw_response)
        print("=" * 50)
        
        # Try to extract JSON
        import re
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            print(f"\n✅ JSON FOUND:")
            print(json_match.group(0))
        else:
            print(f"\n❌ NO JSON FOUND")
            print("Response doesn't contain valid JSON structure")
        
        return raw_response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    debug_single_response()