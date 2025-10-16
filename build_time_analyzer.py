#!/usr/bin/env python3
"""
OpenCV Build Time Analysis - Why Does It Take 20+ Minutes?
This script analyzes the specific reasons for long OpenCV build times in Termux
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path

def run_cmd_timed(cmd, timeout=120):
    """Run command and measure execution time"""
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        end_time = time.time()
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': end_time - start_time,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        end_time = time.time()
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Command timed out',
            'duration': end_time - start_time,
            'returncode': -1
        }

def analyze_why_build_takes_long():
    """Analyze the root causes of long OpenCV build times"""
    print("⏱️  ANALYZING WHY OPENCV BUILD TAKES 20+ MINUTES")
    print("=" * 60)
    
    analysis = {
        'build_process_breakdown': {},
        'bottlenecks': [],
        'hardware_limitations': {},
        'network_factors': {},
        'compilation_factors': {}
    }
    
    # 1. Architecture and Platform Analysis
    print("\n🏗️  BUILD PROCESS BREAKDOWN")
    print("-" * 40)
    
    build_phases = {
        'download': 'Downloading source code and dependencies',
        'configure': 'CMake configuration and dependency detection', 
        'compile': 'Actual compilation of C++ code',
        'link': 'Linking object files into libraries',
        'install': 'Installing compiled libraries'
    }
    
    print("OpenCV build process phases:")
    for phase, description in build_phases.items():
        print(f"  {phase:12} | {description}")
    
    # 2. Hardware Bottlenecks
    print("\n💻 HARDWARE BOTTLENECK ANALYSIS")
    print("-" * 40)
    
    # CPU Analysis
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = f.read()
            
        # Count CPU cores
        cpu_count = cpu_info.count('processor')
        analysis['hardware_limitations']['cpu_cores'] = cpu_count
        print(f"CPU Cores: {cpu_count}")
        
        # Identify CPU type
        if 'ARMv8' in cpu_info or 'aarch64' in cpu_info:
            cpu_type = 'ARM64'
        elif 'ARM' in cpu_info:
            cpu_type = 'ARM32'
        else:
            cpu_type = 'Unknown'
        
        analysis['hardware_limitations']['cpu_type'] = cpu_type
        print(f"CPU Type: {cpu_type}")
        
        # Check for performance cores vs efficiency cores
        if 'big.LITTLE' in cpu_info or 'DynamIQ' in cpu_info:
            print("⚠️  Heterogeneous CPU detected - some cores may be slower")
            analysis['bottlenecks'].append('heterogeneous_cpu_architecture')
        
        # ARM-specific compilation challenges
        if 'ARM' in cpu_type:
            print("🎯 ARM ARCHITECTURE BUILD CHALLENGES:")
            challenges = [
                "Cross-compilation complexity",
                "Limited pre-compiled wheels for ARM",
                "NEON instruction set optimization",
                "Memory alignment requirements",
                "Slower compilation compared to x86"
            ]
            for challenge in challenges:
                print(f"   • {challenge}")
            analysis['bottlenecks'].extend(['arm_compilation', 'limited_arm_wheels'])
    
    except Exception as e:
        print(f"❌ Could not analyze CPU: {e}")
    
    # Memory Analysis
    try:
        with open('/proc/meminfo', 'r') as f:
            mem_info = f.read()
            
        for line in mem_info.split('\n'):
            if 'MemTotal:' in line:
                total_mem_kb = int(line.split()[1])
                total_mem_gb = total_mem_kb / (1024 * 1024)
                analysis['hardware_limitations']['total_memory_gb'] = total_mem_gb
                print(f"Total Memory: {total_mem_gb:.1f} GB")
                
                if total_mem_gb < 2:
                    print("❌ CRITICAL: Low memory will cause VERY slow compilation")
                    print("   • Frequent swapping to storage")
                    print("   • Reduced parallel compilation jobs")
                    print("   • Potential out-of-memory build failures")
                    analysis['bottlenecks'].append('insufficient_memory')
                elif total_mem_gb < 4:
                    print("⚠️  Limited memory - will slow compilation")
                    analysis['bottlenecks'].append('limited_memory')
                else:
                    print("✅ Sufficient memory for compilation")
                break
    except Exception as e:
        print(f"❌ Could not analyze memory: {e}")
    
    # Storage Analysis
    print(f"\n💾 STORAGE PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    # Check if running on internal storage vs SD card
    result = run_cmd_timed("df -h $PREFIX")
    if result['success']:
        lines = result['stdout'].split('\n')
        for line in lines:
            if '/data' in line or 'com.termux' in line:
                parts = line.split()
                if len(parts) >= 4:
                    total_space = parts[1]
                    used_space = parts[2] 
                    available_space = parts[3]
                    usage_percent = parts[4]
                    
                    print(f"Storage: {used_space}/{total_space} used ({usage_percent})")
                    print(f"Available: {available_space}")
                    
                    # Estimate if running on internal storage (faster) or SD card (slower)
                    if 'G' in total_space:
                        total_gb = float(total_space.replace('G', ''))
                        if total_gb > 64:
                            print("🐌 Likely on SD card - SLOW storage will significantly increase build time")
                            analysis['bottlenecks'].append('slow_storage')
                        else:
                            print("⚡ Likely on internal storage - better performance")
                    break
    
    # Test actual storage performance
    print("\n🧪 Testing storage write speed...")
    test_result = run_cmd_timed("dd if=/dev/zero of=/tmp/test_write bs=1M count=10 2>&1")
    if test_result['success']:
        output = test_result['stderr']
        if 'MB/s' in output:
            # Extract speed
            speed_line = [line for line in output.split('\n') if 'MB/s' in line][-1]
            if speed_line:
                print(f"Storage write speed: {speed_line.split()[-2]} MB/s")
                try:
                    speed = float(speed_line.split()[-2])
                    if speed < 10:
                        print("🐌 VERY slow storage - major build time bottleneck")
                        analysis['bottlenecks'].append('very_slow_storage')
                    elif speed < 30:
                        print("⚠️  Slow storage - will increase build time")
                except:
                    pass
        
        # Clean up test file
        run_cmd_timed("rm -f /tmp/test_write")
    
    # 3. Network Factors
    print(f"\n🌐 NETWORK FACTORS ANALYSIS")
    print("-" * 40)
    
    # Test download speed
    print("🧪 Testing download speed...")
    
    # Test with a small file from PyPI
    download_test = run_cmd_timed("curl -o /tmp/test_download -w '%{speed_download}' -s https://pypi.org/simple/ | head -c 1000")
    if download_test['success']:
        # The speed should be in the output
        print("✅ Network connectivity working")
    else:
        print("❌ Network connectivity issues")
        analysis['bottlenecks'].append('network_issues')
    
    # Clean up
    run_cmd_timed("rm -f /tmp/test_download")
    
    # Check if using WiFi vs mobile data
    print("\n📱 Connection type factors:")
    print("   • WiFi: Generally faster, more stable")
    print("   • Mobile data: May have caps, throttling")
    print("   • Download size: OpenCV source ~100MB + dependencies ~200MB")
    
    # 4. Compilation-Specific Factors
    print(f"\n⚙️  COMPILATION-SPECIFIC BOTTLENECKS")
    print("-" * 40)
    
    # Check parallel compilation capability
    cpu_cores = analysis['hardware_limitations'].get('cpu_cores', 1)
    max_parallel_jobs = max(1, cpu_cores - 1)  # Leave one core for system
    
    print(f"Max parallel compilation jobs: {max_parallel_jobs}")
    
    if cpu_cores <= 2:
        print("❌ CRITICAL: Very limited parallelization capability")
        print("   • Single-threaded compilation is EXTREMELY slow")
        print("   • Each OpenCV module compiled sequentially")
        analysis['bottlenecks'].append('limited_parallelization')
    elif cpu_cores <= 4:
        print("⚠️  Limited parallelization - will increase build time")
        analysis['bottlenecks'].append('reduced_parallelization')
    else:
        print("✅ Good parallelization capability")
    
    # Check compilation flags and optimizations
    print(f"\n🏁 Build optimization factors:")
    
    build_optimizations = {
        'BUILD_SHARED_LIBS': 'Build shared libraries (faster linking)',
        'CMAKE_BUILD_TYPE': 'Release vs Debug (Release is faster)',
        'CPU_BASELINE': 'CPU instruction set optimizations',
        'BUILD_PERF_TESTS': 'Skip performance tests (faster build)',
        'BUILD_opencv_apps': 'Skip example applications (faster build)',
        'WITH_GTK': 'Skip GUI dependencies (faster build for headless)'
    }
    
    for flag, description in build_optimizations.items():
        print(f"   • {flag}: {description}")
    
    # 5. Why 20+ Minutes Specifically
    print(f"\n⏰ WHY EXACTLY 20+ MINUTES?")
    print("-" * 40)
    
    time_breakdown = {
        'Download (5-10%)': '1-2 minutes - Source code and dependencies',
        'CMake Configure (5-10%)': '1-2 minutes - Dependency detection, configuration',
        'Compilation (70-80%)': '14-16 minutes - Actual C++ compilation',
        'Linking (10-15%)': '2-3 minutes - Creating shared libraries',
        'Installation (5%)': '1 minute - Copying files to final location'
    }
    
    print("Typical time breakdown for OpenCV compilation:")
    for phase, description in time_breakdown.items():
        print(f"  {phase:20} | {description}")
    
    print(f"\n🎯 PRIMARY TIME CONSUMERS:")
    primary_factors = [
        "C++ compilation is inherently slow (hundreds of source files)",
        "ARM processors are slower than x86 for compilation",
        "Limited memory forces reduced parallelization", 
        "Mobile storage is slower than desktop SSDs",
        "Cross-compilation adds overhead",
        "OpenCV is a LARGE codebase (~500k lines of C++)"
    ]
    
    for factor in primary_factors:
        print(f"   • {factor}")
    
    return analysis

def estimate_build_time_factors():
    """Estimate how different factors affect build time"""
    print(f"\n📊 BUILD TIME ESTIMATION FACTORS")
    print("=" * 50)
    
    # Base time for modern desktop (reference point)
    base_time_minutes = 5  # Modern x86 desktop with SSD
    
    factors = {
        'cpu_architecture': {
            'ARM (mobile)': 3.0,    # ARM is ~3x slower for compilation
            'x86 (desktop)': 1.0    # Reference
        },
        'cpu_cores': {
            '1-2 cores': 4.0,       # Very limited parallelization
            '3-4 cores': 2.5,       # Some parallelization  
            '5-8 cores': 1.5,       # Good parallelization
            '8+ cores': 1.0         # Excellent parallelization
        },
        'memory': {
            '<2GB': 3.0,            # Swapping, reduced parallel jobs
            '2-4GB': 2.0,           # Limited parallel jobs
            '4-8GB': 1.5,           # Adequate
            '8GB+': 1.0             # Optimal
        },
        'storage': {
            'SD card': 2.5,         # Very slow I/O
            'Internal flash': 1.5,  # Moderate I/O
            'High-speed SSD': 1.0   # Fast I/O
        },
        'network': {
            'Slow mobile': 1.5,     # Slow downloads
            'Fast WiFi': 1.2,       # Some download time
            'Cached/Local': 1.0     # No download time
        }
    }
    
    print("Time multiplier factors (vs modern desktop baseline):")
    for category, values in factors.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for condition, multiplier in values.items():
            estimated_time = base_time_minutes * multiplier
            print(f"  {condition:15} | x{multiplier:3.1f} = ~{estimated_time:4.1f} minutes")
    
    # Typical Termux scenario
    print(f"\n🎯 TYPICAL TERMUX SCENARIO ESTIMATE:")
    print("-" * 40)
    
    typical_multipliers = {
        'ARM architecture': 3.0,
        '2-4 CPU cores': 2.5, 
        'Limited memory (2-4GB)': 2.0,
        'Mobile storage': 1.5,
        'Mobile network': 1.2
    }
    
    total_multiplier = 1.0
    for factor, multiplier in typical_multipliers.items():
        total_multiplier *= multiplier
        print(f"  {factor:25} | x{multiplier:3.1f}")
    
    estimated_total = base_time_minutes * total_multiplier
    print(f"\nTotal multiplier: x{total_multiplier:.1f}")
    print(f"Estimated build time: {estimated_total:.0f} minutes")
    print(f"Range: {estimated_total*0.8:.0f}-{estimated_total*1.2:.0f} minutes")
    
    if estimated_total >= 20:
        print(f"✅ This explains the 20+ minute build times!")
    else:
        print(f"🤔 Model suggests shorter time - other factors involved")

def provide_build_time_solutions():
    """Provide specific solutions to reduce build time"""
    print(f"\n💡 SOLUTIONS TO REDUCE BUILD TIME")
    print("=" * 50)
    
    solutions = [
        {
            'category': 'Avoid Compilation Entirely',
            'solutions': [
                ('Use pre-compiled wheels', 'pip install opencv-python-headless', 'Instant'),
                ('Use system packages', 'pkg install opencv python', '1-2 minutes'),
                ('Use compatibility layer', 'Use existing cv2_compat.py', 'Instant')
            ]
        },
        {
            'category': 'Optimize Compilation Process', 
            'solutions': [
                ('Increase swap space', 'Create 2GB swap file', 'Enables more parallel jobs'),
                ('Use minimal build', 'CMAKE_ARGS="-DBUILD_opencv_apps=OFF"', '30% faster'),
                ('Disable unnecessary modules', 'CMAKE_ARGS="-DBUILD_PERF_TESTS=OFF"', '20% faster'),
                ('Use Release build', 'CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"', '10% faster')
            ]
        },
        {
            'category': 'Hardware Optimizations',
            'solutions': [
                ('Close background apps', 'Free up RAM for compilation', '10-20% faster'),
                ('Use cooling', 'Prevent thermal throttling', 'Maintain performance'),
                ('Connect to power', 'Avoid battery-saving mode', 'Full CPU performance')
            ]
        },
        {
            'category': 'Network Optimizations',
            'solutions': [
                ('Use WiFi instead of mobile', 'Faster downloads', '5-10% faster'),
                ('Download during off-peak', 'Less network congestion', '5% faster'),
                ('Use pip cache', 'pip cache dir', 'Reuse downloads')
            ]
        }
    ]
    
    for category_info in solutions:
        category = category_info['category']
        print(f"\n🔧 {category}:")
        print("-" * (len(category) + 5))
        
        for solution, command, benefit in category_info['solutions']:
            print(f"  • {solution:25} | {command:35} | {benefit}")
    
    print(f"\n⭐ RECOMMENDED APPROACH:")
    print("1. Try pre-compiled packages first (opencv-python-headless)")
    print("2. If unavailable, use system packages (pkg install opencv)")
    print("3. If compilation required, optimize build settings")
    print("4. As last resort, use compatibility layer (cv2_compat.py)")
    print("\n💡 Your bot already works with the compatibility layer!")

def main():
    """Main analysis function"""
    print("⏱️  OpenCV Build Time Root Cause Analysis")
    print("=" * 60)
    print("Understanding why OpenCV takes 20+ minutes to build in Termux")
    print("=" * 60)
    
    # Perform analysis
    analysis = analyze_why_build_takes_long()
    estimate_build_time_factors()
    provide_build_time_solutions()
    
    # Summary
    print(f"\n📋 SUMMARY OF FINDINGS")
    print("=" * 30)
    
    bottlenecks = analysis.get('bottlenecks', [])
    
    print("Identified bottlenecks:")
    if not bottlenecks:
        print("  ✅ No major bottlenecks identified")
    else:
        bottleneck_descriptions = {
            'arm_compilation': 'ARM architecture compilation overhead',
            'insufficient_memory': 'Not enough RAM for parallel compilation',
            'limited_memory': 'Limited RAM reduces compilation efficiency',
            'slow_storage': 'Slow storage increases I/O time',
            'very_slow_storage': 'Very slow storage is major bottleneck',
            'limited_parallelization': 'Too few CPU cores for parallel builds',
            'network_issues': 'Network connectivity problems'
        }
        
        for bottleneck in bottlenecks:
            description = bottleneck_descriptions.get(bottleneck, bottleneck)
            print(f"  🚨 {description}")
    
    print(f"\n🎯 ROOT CAUSE: OpenCV compilation is inherently slow on mobile devices")
    print("   • ARM processors + limited resources + large codebase = long build times")
    print("   • 20+ minutes is NORMAL for Termux OpenCV compilation")
    print("   • Use alternatives to avoid compilation entirely")
    
    # Save analysis
    try:
        with open('opencv_build_time_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n📄 Analysis saved to: opencv_build_time_analysis.json")
    except Exception as e:
        print(f"\n❌ Could not save analysis: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted")
    except Exception as e:
        print(f"\n\n🚨 Analysis failed: {e}")
        import traceback
        traceback.print_exc()