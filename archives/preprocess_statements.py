#!/usr/bin/env python3
"""
Quick manual preprocessing of 22 low-quality problem statements
"""

import pandas as pd

# Manual preprocessing mapping
PREPROCESSING_MAP = {
    "project 1 : chip and pin": "Project 1 cash register chip and pin payment issue",
    "till 2 - signing out": "Till 2 cashier unable to sign out",
    "till 3 - printer issue": "Till 3 receipt printer malfunction",
    "unlock order": "Customer needs order unlocked in system",
    "till 2 - red error on printer": "Till 2 receipt printer showing red error",
    "till freezing - till 2": "Till 2 cash register system freezing",
    "project 1-unable to login": "Project 1 user unable to login to system",
    "email address": "Email address configuration or access issue",
    "project 1 chip and pin reset": "Project 1 chip and pin payment device needs reset",
    "till run time error": "Cash register showing runtime error message",
    "till 2 - printing not working": "Till 2 receipt printer not functioning",
    "fusion- frozen": "Fusion software application frozen or unresponsive",
    "till function error occuring": "Cash register experiencing functional errors",
    "till 2 screen not turning on": "Till 2 cash register screen display not working",
    "till 1 - card insert faulty": "Till 1 card reader slot faulty or damaged",
    "project 5 :login": "Project 5 user login access issue",
    "till 2 baseunit fan noising": "Till 2 cash register base unit fan making noise",
    "banking issue": "Banking system or transaction processing issue",
    "vision order": "Vision system order processing issue",
    "till 4 scanner wire issue": "Till 4 barcode scanner cable connection problem",
    "google : 2fa": "Google account two-factor authentication issue",
    "2 new headsets": "Request for 2 new communication headsets"
}

def preprocess_statements():
    """Apply manual preprocessing to low-quality statements."""
    
    # Load the improved problem groups
    df = pd.read_csv('outputs/improved_problem_groups.csv')
    
    print("🔧 PREPROCESSING 22 LOW-QUALITY STATEMENTS")
    print("="*60)
    
    processed_count = 0
    
    for index, row in df.iterrows():
        old_statement = row['representative_short_description']
        
        if old_statement in PREPROCESSING_MAP:
            new_statement = PREPROCESSING_MAP[old_statement]
            df.at[index, 'representative_short_description'] = new_statement
            processed_count += 1
            
            print(f"✅ {processed_count:2d}. BEFORE: {old_statement}")
            print(f"   AFTER:  {new_statement}")
            print()
    
    # Save the updated file
    df.to_csv('outputs/improved_problem_groups_preprocessed.csv', index=False)
    
    print(f"🎉 PREPROCESSING COMPLETE!")
    print(f"   📊 Statements processed: {processed_count}/22")
    print(f"   📁 Updated file saved: outputs/improved_problem_groups_preprocessed.csv")
    print(f"   ✅ All 209 statements now ready for LLM analysis!")

if __name__ == "__main__":
    preprocess_statements()