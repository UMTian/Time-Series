#!/usr/bin/env python3
"""
Simple script to fix the old function call in automated_optimization_framework.py
"""

def fix_function_call():
    """Fix the old function call to use the enhanced parameters"""
    
    # Read the file
    with open('automated_optimization_framework.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the old function call
    old_call = "configs = self.get_industry_parameters()"
    new_call = "configs = self.get_advanced_parameters()"
    
    if old_call in content:
        content = content.replace(old_call, new_call)
        print("✅ Fixed function call from get_industry_parameters to get_advanced_parameters")
    else:
        print("❌ Old function call not found")
        return False
    
    # Write the fixed content back
    with open('automated_optimization_framework.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ File updated successfully!")
    return True

if __name__ == "__main__":
    fix_function_call()
