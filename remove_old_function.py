#!/usr/bin/env python3
"""
Script to remove the old get_industry_parameters function
"""

def remove_old_function():
    """Remove the old get_industry_parameters function"""
    
    # Read the file
    with open('automated_optimization_framework.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the start and end of the old function
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if 'def get_industry_parameters(self):' in line:
            start_line = i
            print(f"Found start of old function at line {i+1}")
        elif start_line is not None and line.strip() == '' and i > start_line + 5:
            # Look for the next function definition
            if 'def ' in line and 'self' in line:
                end_line = i
                break
    
    if start_line is not None and end_line is not None:
        # Remove the old function
        lines = lines[:start_line] + lines[end_line:]
        
        # Write the updated content back
        with open('automated_optimization_framework.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"✅ Removed old function from line {start_line+1} to {end_line}")
        return True
    else:
        print("❌ Could not find the old function boundaries")
        return False

if __name__ == "__main__":
    remove_old_function()
