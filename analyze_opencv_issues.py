#!/usr/bin/env python3
"""
OpenCV Root Cause Master Analysis
Comprehensive analysis of OpenCV installation failures and solutions
"""

import sys
import os
import subprocess
import json
import platform

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔬 {title}")
    print('='*60)

def main():
    print_header("OPENCV ROOT CAUSE ANALYSIS SUITE")
    
    print("This suite will analyze your Termux environment to understand:")
    print("1. Why OpenCV installation fails")
    print("2. Why compilation takes 20+ minutes")
    print("3. What the root causes are")
    print("4. Which solutions will work for your specific setup")
    
    print(f"\n📋 Available Analysis Tools:")
    print("1. 🔧 Root Cause Analyzer - Comprehensive system analysis")
    print("2. 🏗️  Installation Structure Analyzer - Deep dive into installation process")
    print("3. ⏱️  Build Time Analyzer - Why compilation takes 20+ minutes")
    print("4. 🧪 Quick Test - Simple functionality test")
    print("5. 📊 Run All Analyses")
    
    try:
        choice = input(f"\nSelect analysis (1-5): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice == '1':
        print_header("RUNNING ROOT CAUSE ANALYZER")
        run_analysis_script('opencv_root_cause_analyzer.py')
    elif choice == '2':
        print_header("RUNNING INSTALLATION STRUCTURE ANALYZER")
        run_analysis_script('opencv_structure_analyzer.py')
    elif choice == '3':
        print_header("RUNNING BUILD TIME ANALYZER")
        run_analysis_script('build_time_analyzer.py')
    elif choice == '4':
        print_header("RUNNING QUICK TEST")
        run_analysis_script('test_opencv.py')
    elif choice == '5':
        print_header("RUNNING COMPLETE ANALYSIS SUITE")
        run_all_analyses()
    else:
        print("Invalid choice. Exiting...")
        return

def run_analysis_script(script_name):
    """Run an analysis script and handle errors"""
    if not os.path.exists(script_name):
        print(f"❌ {script_name} not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run {script_name}: {e}")
        return False

def run_all_analyses():
    """Run all analysis scripts in sequence"""
    scripts = [
        ('test_opencv.py', 'Quick OpenCV Test'),
        ('opencv_root_cause_analyzer.py', 'Root Cause Analysis'),
        ('opencv_structure_analyzer.py', 'Installation Structure Analysis'), 
        ('build_time_analyzer.py', 'Build Time Analysis')
    ]
    
    results = {}
    
    for script, description in scripts:
        print(f"\n{'─'*40}")
        print(f"🔍 Running: {description}")
        print('─'*40)
        
        if os.path.exists(script):
            success = run_analysis_script(script)
            results[script] = success
            print(f"{'✅' if success else '❌'} {description}: {'Completed' if success else 'Failed'}")
        else:
            print(f"⚠️  {script} not found - skipping")
            results[script] = False
    
    # Summary
    print_header("ANALYSIS SUITE COMPLETE")
    
    print("📊 Results Summary:")
    for script, success in results.items():
        status = "✅ Completed" if success else "❌ Failed"
        print(f"  {script:30} | {status}")
    
    # Check for generated reports
    report_files = [
        'opencv_analysis_report.json',
        'opencv_structure_analysis.json', 
        'opencv_build_time_analysis.json'
    ]
    
    available_reports = [f for f in report_files if os.path.exists(f)]
    
    if available_reports:
        print(f"\n📄 Generated Reports:")
        for report in available_reports:
            size = os.path.getsize(report)
            print(f"  📋 {report} ({size} bytes)")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Review the analysis outputs above")
    print("2. Check generated JSON reports for detailed data")
    print("3. Follow the recommended solutions")
    print("4. Test your bot with: python bot_run.py")

if __name__ == "__main__":
    main()