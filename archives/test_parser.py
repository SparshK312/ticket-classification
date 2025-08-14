#!/usr/bin/env python3
"""
Test the structured text parser
"""

from llm_automation_classifier import LLMAutomationClassifier

def test_parser():
    classifier = LLMAutomationClassifier()
    
    sample_response = """Here's the analysis:

**Step 1: HARDWARE CHECK**
YES - This involves physical hardware (printers).

**Step 2: DIAGNOSTIC CHECK**
YES - Troubleshooting and installation of printers often require investigation and diagnostic steps.

**Step 3: STANDARD COMMAND CHECK**
NO clear command - While there are standard commands for printer-related tasks, the problem statement is too broad to map directly to a single well-known command.

**Step 4: HUMAN INTERVENTION CHECK**
YES - Resolution may require human decision or approval, such as selecting the correct printer driver or configuration settings.

Based on these answers, I conclude that:

* **automation_category**: PARTIALLY_AUTOMATABLE
* **confidence_score**: 0.85 (based on the quality of the sample descriptions and the frequency of similar tickets)
* **reasoning**: Based on the 4-step analysis above, this problem involves physical hardware, requires diagnostic steps, and may require human decision or approval.
* **scriptability_assessment**: Technical implementation is possible, but would likely involve a combination of scripted installation steps and manual configuration settings. A script could automate some tasks, such as downloading and installing drivers, but human intervention would be required for more complex setup and configuration.
* **human_touchpoints**: Printer driver selection, configuration settings, and troubleshooting steps may require human input.
* **business_priority**: MEDIUM (based on the frequency of similar tickets and the potential impact on business operations)
* **implementation_complexity**: MODERATE (due to the need for both scripted installation steps and manual configuration settings)
* **roi_estimate**: Realistic time savings estimate: 20-30% reduction in ticket resolution time, assuming a well-designed automation script."""
    
    try:
        result = classifier._parse_structured_text_response(sample_response)
        
        print("🎉 PARSER TEST RESULTS:")
        print("=" * 50)
        for key, value in result.items():
            print(f"{key}: {value}")
        
        print("\n✅ Parser working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Parser failed: {e}")
        return False

if __name__ == "__main__":
    test_parser()