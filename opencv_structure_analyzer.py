#!/usr/bin/env python3
"""
OpenCV Installation Structure Analyzer
Deep dive into how OpenCV should be installed and what goes wrong
"""

import sys
import os
import subprocess
import json
import tempfile
from pathlib import Path

def run_cmd(cmd, timeout=60):
    """Run command with proper error handling"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def analyze_opencv_wheel_structure():
    """Analyze OpenCV wheel structure and what should be installed"""
    print("🔍 ANALYZING OPENCV WHEEL STRUCTURE")
    print("=" * 50)
    
    analysis = {
        'available_wheels': {},
        'wheel_contents': {},
        'size_analysis': {},
        'platform_compatibility': {}
    }
    
    # Check available OpenCV packages
    opencv_packages = [
        'opencv-python',
        'opencv-python-headless',
        'opencv-contrib-python', 
        'opencv-contrib-python-headless'
    ]
    
    for package in opencv_packages:
        print(f"\n📦 Analyzing {package}...")
        
        # Get package information
        success, stdout, stderr = run_cmd(f"pip index versions {package}")
        if success:
            analysis['available_wheels'][package] = {
                'available': True,
                'versions': stdout.strip()
            }
            print(f"✅ {package} is available")
            
            # Try to get wheel file info without installing
            success2, stdout2, stderr2 = run_cmd(f"pip download --no-deps --only-binary=:all: {package} 2>&1")
            if success2:
                # Parse download output to find wheel filename
                lines = stdout2.split('\n')
                wheel_file = None
                for line in lines:
                    if '.whl' in line and 'Downloading' in line:
                        wheel_file = line.split()[-1]
                        break
                
                if wheel_file:
                    analysis['available_wheels'][package]['wheel_file'] = wheel_file
                    print(f"  📁 Wheel file: {wheel_file}")
                    
                    # Analyze wheel filename for platform compatibility
                    parts = wheel_file.split('-')
                    if len(parts) >= 5:
                        version = parts[1]
                        python_tag = parts[2]
                        abi_tag = parts[3] 
                        platform_tag = parts[4].replace('.whl', '')
                        
                        analysis['platform_compatibility'][package] = {
                            'version': version,
                            'python_tag': python_tag,
                            'abi_tag': abi_tag,
                            'platform_tag': platform_tag
                        }
                        
                        print(f"  🏷️  Version: {version}")
                        print(f"  🐍 Python: {python_tag}")
                        print(f"  🔧 ABI: {abi_tag}")
                        print(f"  💻 Platform: {platform_tag}")
                        
                        # Check platform compatibility
                        current_platform = get_current_platform_tag()
                        if platform_tag == 'any' or current_platform in platform_tag:
                            print(f"  ✅ Platform compatible")
                        else:
                            print(f"  ❌ Platform mismatch: need {platform_tag}, have {current_platform}")
            else:
                print(f"  ⚠️  Could not download wheel: {stderr2}")
        else:
            analysis['available_wheels'][package] = {
                'available': False,
                'error': stderr
            }
            print(f"❌ {package} not available: {stderr}")
    
    return analysis

def get_current_platform_tag():
    """Get current platform tag for wheel compatibility"""
    try:
        success, stdout, stderr = run_cmd("python -c \"import sysconfig; print(sysconfig.get_platform())\"")
        if success:
            return stdout.strip()
    except:
        pass
    
    # Fallback
    import platform
    machine = platform.machine().lower()
    if 'aarch64' in machine or 'arm64' in machine:
        return 'linux_aarch64'
    elif 'arm' in machine:
        return 'linux_armv7l'
    else:
        return 'unknown'

def analyze_installation_process():
    """Trace through the actual installation process step by step"""
    print("\n🔄 ANALYZING INSTALLATION PROCESS")
    print("=" * 50)
    
    analysis = {
        'installation_steps': [],
        'failure_points': [],
        'resource_usage': {},
        'dependencies': {}
    }
    
    # Step 1: pip download analysis
    print("\n1️⃣ Testing pip download process...")
    
    test_package = 'opencv-python-headless'
    success, stdout, stderr = run_cmd(f"pip download --no-deps --no-binary=:none: {test_package} --dry-run")
    
    step1 = {
        'step': 'pip_download',
        'success': success,
        'output': stdout[:500] if stdout else '',
        'error': stderr[:500] if stderr else ''
    }
    analysis['installation_steps'].append(step1)
    
    if success:
        print("✅ Pip download process works")
    else:
        print(f"❌ Pip download fails: {stderr[:200]}")
        analysis['failure_points'].append('pip_download')
    
    # Step 2: Dependency resolution
    print("\n2️⃣ Testing dependency resolution...")
    
    success, stdout, stderr = run_cmd(f"pip install --dry-run {test_package}")
    
    step2 = {
        'step': 'dependency_resolution',
        'success': success,
        'output': stdout[:500] if stdout else '',
        'error': stderr[:500] if stderr else ''
    }
    analysis['installation_steps'].append(step2)
    
    if success:
        print("✅ Dependency resolution works")
        # Parse dependencies
        if 'numpy' in stdout.lower():
            print("  📦 Requires numpy")
        if 'pillow' in stdout.lower():
            print("  📦 Requires pillow")
    else:
        print(f"❌ Dependency resolution fails: {stderr[:200]}")
        analysis['failure_points'].append('dependency_resolution')
    
    # Step 3: Check wheel vs source installation
    print("\n3️⃣ Testing wheel vs source installation preference...")
    
    # Check if wheels are preferred
    success, stdout, stderr = run_cmd(f"pip install --only-binary=:all: --dry-run {test_package}")
    
    if success:
        print("✅ Binary wheel installation preferred")
        analysis['installation_steps'].append({
            'step': 'wheel_installation',
            'success': True,
            'method': 'binary_wheel'
        })
    else:
        print("⚠️  Binary wheel not available, would compile from source")
        analysis['installation_steps'].append({
            'step': 'wheel_installation', 
            'success': False,
            'method': 'source_compilation',
            'error': stderr[:200]
        })
        analysis['failure_points'].append('wheel_availability')
        
        # Check compilation requirements
        print("\n🔨 Checking compilation requirements...")
        analyze_compilation_requirements(analysis)
    
    return analysis

def analyze_compilation_requirements(analysis):
    """Analyze what's needed for source compilation"""
    print("\n4️⃣ Analyzing compilation requirements...")
    
    compilation_deps = {
        'build_tools': ['gcc', 'g++', 'clang', 'make', 'cmake', 'ninja'],
        'dev_packages': ['python-dev', 'python3-dev'],
        'image_libraries': ['libjpeg-dev', 'libpng-dev', 'libtiff-dev'],
        'system_libraries': ['libopencv-dev', 'libatlas-base-dev']
    }
    
    missing_deps = []
    
    for category, deps in compilation_deps.items():
        print(f"\n📋 Checking {category}...")
        for dep in deps:
            # Check if available in Termux
            success, stdout, stderr = run_cmd(f"pkg show {dep}")
            if success:
                # Check if installed
                success2, stdout2, stderr2 = run_cmd(f"pkg list-installed | grep {dep}")
                if success2 and stdout2:
                    print(f"  ✅ {dep}: Installed")
                else:
                    print(f"  ⚠️  {dep}: Available but not installed")
                    missing_deps.append(dep)
            else:
                print(f"  ❌ {dep}: Not available in Termux")
                missing_deps.append(dep)
    
    analysis['dependencies']['missing_compilation_deps'] = missing_deps
    
    if missing_deps:
        print(f"\n🚨 Missing dependencies for compilation: {len(missing_deps)} items")
        analysis['failure_points'].append('missing_compilation_dependencies')
    else:
        print("\n✅ All compilation dependencies available")
    
    # Check available memory and disk space for compilation
    print("\n💾 Checking resources for compilation...")
    
    # Memory check
    try:
        with open('/proc/meminfo', 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'MemAvailable' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024 * 1024)
                    analysis['resource_usage']['available_memory_gb'] = mem_gb
                    
                    if mem_gb < 1.0:
                        print(f"  ❌ Low memory: {mem_gb:.1f}GB (need 1GB+ for compilation)")
                        analysis['failure_points'].append('insufficient_memory')
                    else:
                        print(f"  ✅ Sufficient memory: {mem_gb:.1f}GB")
                    break
    except:
        print("  ❓ Could not check memory")
    
    # Disk space check
    success, stdout, stderr = run_cmd("df -h $PREFIX")
    if success:
        lines = stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            available = parts[3] if len(parts) > 3 else "unknown"
            print(f"  💾 Available storage: {available}")
            
            # Check if sufficient (rough estimate)
            if 'M' in available and int(available.replace('M', '')) < 500:
                print(f"  ❌ Low storage: {available} (need 1GB+ for compilation)")
                analysis['failure_points'].append('insufficient_storage')
            elif 'G' in available:
                gb = float(available.replace('G', ''))
                if gb < 1.0:
                    print(f"  ❌ Low storage: {available} (need 1GB+ for compilation)")
                    analysis['failure_points'].append('insufficient_storage')
                else:
                    print(f"  ✅ Sufficient storage: {available}")

def analyze_runtime_dependencies():
    """Analyze what OpenCV needs at runtime"""
    print("\n🏃 ANALYZING RUNTIME DEPENDENCIES")
    print("=" * 50)
    
    analysis = {
        'python_modules': {},
        'system_libraries': {},
        'missing_runtime_deps': []
    }
    
    # Check Python module dependencies
    essential_python_modules = ['numpy', 'typing_extensions']
    optional_python_modules = ['pillow', 'matplotlib', 'scipy']
    
    print("\n📦 Essential Python modules:")
    for module in essential_python_modules:
        try:
            exec(f"import {module}")
            print(f"  ✅ {module}: Available")
            analysis['python_modules'][module] = 'available'
        except ImportError:
            print(f"  ❌ {module}: Missing - REQUIRED")
            analysis['python_modules'][module] = 'missing'
            analysis['missing_runtime_deps'].append(module)
    
    print("\n📦 Optional Python modules:")
    for module in optional_python_modules:
        try:
            exec(f"import {module}")
            print(f"  ✅ {module}: Available")
            analysis['python_modules'][module] = 'available'
        except ImportError:
            print(f"  ⚠️  {module}: Missing - optional")
            analysis['python_modules'][module] = 'missing'
    
    # Check system libraries
    print("\n🔧 System libraries:")
    
    # Try to find important libraries
    lib_search_paths = [
        '$PREFIX/lib',
        '/system/lib64',
        '/system/lib'
    ]
    
    important_libs = [
        'libc.so',
        'libm.so', 
        'libz.so',
        'libpng.so',
        'libjpeg.so'
    ]
    
    for lib in important_libs:
        found = False
        for search_path in lib_search_paths:
            success, stdout, stderr = run_cmd(f"find {search_path} -name '{lib}*' 2>/dev/null | head -1")
            if success and stdout.strip():
                print(f"  ✅ {lib}: Found at {stdout.strip()}")
                analysis['system_libraries'][lib] = stdout.strip()
                found = True
                break
        
        if not found:
            print(f"  ❌ {lib}: Not found")
            analysis['system_libraries'][lib] = None
    
    return analysis

def generate_installation_structure_report(wheel_analysis, process_analysis, runtime_analysis):
    """Generate comprehensive report about OpenCV installation structure"""
    print("\n📊 INSTALLATION STRUCTURE ANALYSIS REPORT")
    print("=" * 60)
    
    # Identify primary failure points
    all_failures = process_analysis.get('failure_points', [])
    
    print("\n🎯 PRIMARY FAILURE POINTS:")
    if not all_failures:
        print("✅ No critical failure points identified")
    else:
        failure_categories = {
            'pip_download': 'Network/Repository Issues',
            'dependency_resolution': 'Dependency Conflicts', 
            'wheel_availability': 'No Compatible Wheels',
            'missing_compilation_dependencies': 'Missing Build Tools',
            'insufficient_memory': 'Resource Constraints',
            'insufficient_storage': 'Resource Constraints'
        }
        
        for failure in all_failures:
            category = failure_categories.get(failure, 'Unknown Issue')
            print(f"  🚨 {failure}: {category}")
    
    # Analyze installation paths
    print("\n🛤️  INSTALLATION PATH ANALYSIS:")
    
    path_options = []
    
    # Option 1: Pre-compiled wheels
    compatible_wheels = []
    for package, info in wheel_analysis.get('available_wheels', {}).items():
        if info.get('available'):
            platform_info = wheel_analysis.get('platform_compatibility', {}).get(package, {})
            if platform_info:
                compatible_wheels.append(package)
    
    if compatible_wheels:
        path_options.append({
            'method': 'Pre-compiled Wheels',
            'feasibility': 'HIGH' if 'wheel_availability' not in all_failures else 'LOW',
            'packages': compatible_wheels,
            'description': 'Download and install pre-built OpenCV wheels'
        })
    
    # Option 2: System packages
    path_options.append({
        'method': 'System Packages (pkg)',
        'feasibility': 'MEDIUM',
        'packages': ['opencv'],
        'description': 'Use Termux repository OpenCV package'
    })
    
    # Option 3: Source compilation
    compilation_feasible = 'insufficient_memory' not in all_failures and 'insufficient_storage' not in all_failures
    path_options.append({
        'method': 'Source Compilation',
        'feasibility': 'HIGH' if compilation_feasible and 'missing_compilation_dependencies' not in all_failures else 'LOW',
        'packages': ['opencv-python-headless'],
        'description': 'Compile OpenCV from source code'
    })
    
    # Option 4: Compatibility layer
    path_options.append({
        'method': 'Compatibility Layer',
        'feasibility': 'HIGH',
        'packages': ['cv2_compat.py'],
        'description': 'Use pure Python OpenCV replacement'
    })
    
    for i, option in enumerate(path_options, 1):
        print(f"\n{i}. {option['method']} - Feasibility: {option['feasibility']}")
        print(f"   📦 Packages: {', '.join(option['packages'])}")
        print(f"   📝 {option['description']}")
    
    # Specific recommendations based on analysis
    print("\n💡 SPECIFIC RECOMMENDATIONS:")
    
    if 'wheel_availability' in all_failures:
        print("1. 🎯 NO COMPATIBLE WHEELS AVAILABLE")
        print("   → Your ARM/Android platform lacks pre-built wheels")
        print("   → Must use system packages or compile from source")
        print("   → Recommended: Try 'pkg install opencv python' first")
    
    if 'missing_compilation_dependencies' in all_failures:
        print("2. 🔧 MISSING COMPILATION TOOLS") 
        print("   → Install build dependencies: pkg install clang cmake ninja")
        print("   → Or use compatibility layer to avoid compilation")
    
    if 'insufficient_memory' in all_failures or 'insufficient_storage' in all_failures:
        print("3. 💾 RESOURCE CONSTRAINTS")
        print("   → Create swap file for more memory")
        print("   → Clear storage space")
        print("   → Use compatibility layer as lightweight alternative")
    
    missing_runtime = runtime_analysis.get('missing_runtime_deps', [])
    if missing_runtime:
        print(f"4. 📦 MISSING RUNTIME DEPENDENCIES")
        print(f"   → Install: pip install {' '.join(missing_runtime)}")
    
    # Final recommendation
    print("\n🎯 RECOMMENDED INSTALLATION STRATEGY:")
    
    if not all_failures:
        print("✅ Standard installation should work:")
        print("   pip install opencv-python-headless")
    elif len(all_failures) == 1 and all_failures[0] == 'wheel_availability':
        print("🔧 Try system packages first:")
        print("   pkg install opencv python")
        print("   If that fails: use compatibility layer")
    elif 'insufficient_memory' in all_failures or 'insufficient_storage' in all_failures:
        print("💡 Use compatibility layer (immediate solution):")
        print("   Your bot already includes cv2_compat.py")
        print("   No installation needed - works immediately")
    else:
        print("🚨 Multiple issues detected:")
        print("   1. Fix missing dependencies")
        print("   2. Try system packages")
        print("   3. Fall back to compatibility layer")
    
    return {
        'failure_points': all_failures,
        'path_options': path_options,
        'recommended_strategy': path_options[0] if path_options else None
    }

def main():
    """Main analysis function"""
    print("🔬 OpenCV Installation Structure Deep Analysis")
    print("=" * 60)
    print("Analyzing the complete OpenCV installation pipeline...")
    print("=" * 60)
    
    # Perform detailed analyses
    wheel_analysis = analyze_opencv_wheel_structure()
    process_analysis = analyze_installation_process()
    runtime_analysis = analyze_runtime_dependencies()
    
    # Generate comprehensive report
    report = generate_installation_structure_report(
        wheel_analysis, process_analysis, runtime_analysis
    )
    
    # Save detailed analysis
    full_analysis = {
        'wheel_structure': wheel_analysis,
        'installation_process': process_analysis,
        'runtime_dependencies': runtime_analysis,
        'final_report': report
    }
    
    try:
        with open('opencv_structure_analysis.json', 'w') as f:
            json.dump(full_analysis, f, indent=2, default=str)
        print(f"\n📄 Detailed analysis saved to: opencv_structure_analysis.json")
    except Exception as e:
        print(f"\n❌ Could not save analysis: {e}")
    
    print(f"\n🏁 ANALYSIS COMPLETE")
    print("Use this information to understand exactly why OpenCV installation fails")
    print("and which alternative approach will work best for your system.")
    
    return len(report.get('failure_points', [])) == 0

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted")
    except Exception as e:
        print(f"\n\n🚨 Analysis failed: {e}")
        import traceback
        traceback.print_exc()