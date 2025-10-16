#!/usr/bin/env python3
"""
OpenCV Installation Root Cause Analyzer for Termux
This script performs deep analysis of why OpenCV fails to install or work properly.
"""

import sys
import os
import subprocess
import platform
import importlib.util
import json
from pathlib import Path

def run_command_safe(cmd, timeout=30):
    """Run command safely with timeout and error handling"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, 
            timeout=timeout, cwd=os.getcwd()
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'stdout': '', 'stderr': 'Command timed out', 'returncode': -1}
    except Exception as e:
        return {'success': False, 'stdout': '', 'stderr': str(e), 'returncode': -1}

def print_section(title, level=1):
    """Print formatted section headers"""
    if level == 1:
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print('='*60)
    elif level == 2:
        print(f"\n{'─'*40}")
        print(f"📋 {title}")
        print('─'*40)
    else:
        print(f"\n• {title}")

def analyze_system_environment():
    """Analyze the system environment for OpenCV compatibility"""
    print_section("SYSTEM ENVIRONMENT ANALYSIS", 1)
    
    analysis = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'architecture': platform.machine(),
        'is_termux': False,
        'available_memory': None,
        'available_storage': None,
        'environment_vars': {},
        'issues': []
    }
    
    # Check if running in Termux
    termux_indicators = [
        '/data/data/com.termux',
        '/system/bin/getprop'
    ]
    
    for indicator in termux_indicators:
        if os.path.exists(indicator):
            analysis['is_termux'] = True
            print("✅ Confirmed Termux environment")
            break
    
    if not analysis['is_termux']:
        print("❌ Not running in Termux - this analysis is Termux-specific")
        analysis['issues'].append("Not running in Termux environment")
    
    # Check architecture compatibility
    arch = platform.machine().lower()
    print(f"Architecture: {arch}")
    
    compatible_archs = ['aarch64', 'arm64', 'armv7l', 'armv8']
    if not any(a in arch for a in compatible_archs):
        analysis['issues'].append(f"Potentially incompatible architecture: {arch}")
        print(f"⚠️  Architecture {arch} may have limited OpenCV support")
    else:
        print(f"✅ Architecture {arch} is compatible with OpenCV")
    
    # Check Python version compatibility
    version_info = sys.version_info
    if version_info.major != 3 or version_info.minor < 8:
        analysis['issues'].append(f"Python {version_info.major}.{version_info.minor} may have limited OpenCV wheel support")
        print(f"⚠️  Python {version_info.major}.{version_info.minor} - OpenCV wheels prefer Python 3.8+")
    else:
        print(f"✅ Python {version_info.major}.{version_info.minor} is compatible")
    
    # Check memory availability
    try:
        result = run_command_safe("cat /proc/meminfo | grep MemAvailable")
        if result['success']:
            mem_line = result['stdout'].strip()
            if mem_line:
                mem_kb = int(mem_line.split()[1])
                mem_mb = mem_kb // 1024
                analysis['available_memory'] = mem_mb
                print(f"Available memory: {mem_mb} MB")
                
                if mem_mb < 512:
                    analysis['issues'].append(f"Low memory: {mem_mb} MB (OpenCV compilation needs 1GB+)")
                    print(f"❌ Low memory: {mem_mb} MB - OpenCV compilation typically needs 1GB+")
                elif mem_mb < 1024:
                    print(f"⚠️  Limited memory: {mem_mb} MB - may struggle with OpenCV compilation")
                else:
                    print(f"✅ Sufficient memory: {mem_mb} MB")
    except:
        print("❌ Could not check memory availability")
    
    # Check storage space
    try:
        result = run_command_safe("df -h $PREFIX")
        if result['success']:
            lines = result['stdout'].strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                available = parts[3] if len(parts) > 3 else "unknown"
                analysis['available_storage'] = available
                print(f"Available storage: {available}")
                
                # Parse size (rough estimation)
                if 'G' in available:
                    size_gb = float(available.replace('G', ''))
                    if size_gb < 1:
                        analysis['issues'].append(f"Low storage: {available} (OpenCV needs 1GB+)")
                        print(f"❌ Low storage: {available} - OpenCV installation needs 1GB+")
                    else:
                        print(f"✅ Sufficient storage: {available}")
                elif 'M' in available:
                    analysis['issues'].append(f"Very low storage: {available}")
                    print(f"❌ Very low storage: {available}")
    except:
        print("❌ Could not check storage availability")
    
    # Check important environment variables
    env_vars = ['PREFIX', 'LD_LIBRARY_PATH', 'PKG_CONFIG_PATH', 'CC', 'CXX', 'CFLAGS', 'CXXFLAGS']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        analysis['environment_vars'][var] = value
        if value != 'Not set':
            print(f"Environment {var}: {value}")
    
    return analysis

def analyze_python_environment():
    """Analyze Python package management and pip configuration"""
    print_section("PYTHON ENVIRONMENT ANALYSIS", 1)
    
    analysis = {
        'pip_version': None,
        'pip_config': {},
        'site_packages': [],
        'installed_packages': {},
        'wheel_support': False,
        'compilation_tools': {},
        'issues': []
    }
    
    # Check pip version
    result = run_command_safe("pip --version")
    if result['success']:
        analysis['pip_version'] = result['stdout'].strip()
        print(f"✅ Pip: {analysis['pip_version']}")
    else:
        analysis['issues'].append("Pip not available or not working")
        print("❌ Pip not available")
        return analysis
    
    # Check wheel support
    result = run_command_safe("pip show wheel")
    if result['success']:
        analysis['wheel_support'] = True
        print("✅ Wheel support available")
    else:
        print("⚠️  Wheel package not installed - may need to compile from source")
        analysis['issues'].append("Wheel package not installed")
    
    # Check compilation tools
    tools = {
        'gcc': 'gcc --version',
        'g++': 'g++ --version', 
        'clang': 'clang --version',
        'make': 'make --version',
        'cmake': 'cmake --version',
        'ninja': 'ninja --version',
        'pkg-config': 'pkg-config --version'
    }
    
    for tool, cmd in tools.items():
        result = run_command_safe(cmd)
        if result['success']:
            version = result['stdout'].split('\n')[0]
            analysis['compilation_tools'][tool] = version
            print(f"✅ {tool}: {version}")
        else:
            analysis['compilation_tools'][tool] = None
            print(f"❌ {tool}: Not available")
    
    # Check critical compilation tools
    if not analysis['compilation_tools'].get('gcc') and not analysis['compilation_tools'].get('clang'):
        analysis['issues'].append("No C compiler available (gcc or clang)")
        print("🚨 CRITICAL: No C compiler found - OpenCV compilation will fail")
    
    if not analysis['compilation_tools'].get('cmake'):
        analysis['issues'].append("CMake not available - required for OpenCV compilation")
        print("🚨 CRITICAL: CMake not found - required for OpenCV compilation")
    
    # Check installed packages related to OpenCV
    opencv_related = ['opencv-python', 'opencv-contrib-python', 'opencv-python-headless', 
                     'opencv-contrib-python-headless', 'numpy', 'pillow', 'scipy']
    
    for package in opencv_related:
        result = run_command_safe(f"pip show {package}")
        if result['success']:
            # Parse package info
            info = {}
            for line in result['stdout'].split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
            analysis['installed_packages'][package] = info
            print(f"✅ {package}: {info.get('Version', 'unknown version')}")
        else:
            analysis['installed_packages'][package] = None
    
    return analysis

def analyze_opencv_installation_attempts():
    """Analyze previous OpenCV installation attempts and failures"""
    print_section("OPENCV INSTALLATION HISTORY ANALYSIS", 1)
    
    analysis = {
        'pip_cache': [],
        'build_logs': [],
        'failed_wheels': [],
        'available_wheels': [],
        'installation_errors': [],
        'issues': []
    }
    
    # Check pip cache for OpenCV
    result = run_command_safe("pip cache list | grep opencv")
    if result['success'] and result['stdout']:
        analysis['pip_cache'] = result['stdout'].strip().split('\n')
        print(f"Found {len(analysis['pip_cache'])} OpenCV entries in pip cache")
        for entry in analysis['pip_cache']:
            print(f"  📦 {entry}")
    else:
        print("No OpenCV entries found in pip cache")
    
    # Check for available wheels
    print_section("Available OpenCV Wheels", 2)
    
    opencv_packages = [
        'opencv-python',
        'opencv-python-headless', 
        'opencv-contrib-python',
        'opencv-contrib-python-headless'
    ]
    
    for package in opencv_packages:
        print(f"\n🔍 Checking {package}...")
        result = run_command_safe(f"pip index versions {package}")
        if result['success']:
            print(f"✅ {package} is available")
            # Try to get more specific wheel info
            result2 = run_command_safe(f"pip debug --verbose | grep {package}")
            if result2['success'] and result2['stdout']:
                analysis['available_wheels'].append(f"{package}: {result2['stdout'].strip()}")
        else:
            print(f"❌ {package} not found or unavailable")
            analysis['issues'].append(f"{package} not available from pip index")
    
    # Simulate installation to see what would happen
    print_section("Installation Simulation", 2)
    
    for package in ['opencv-python-headless']:  # Test the most likely to work
        print(f"\n🧪 Simulating {package} installation...")
        result = run_command_safe(f"pip install --dry-run --no-deps {package}")
        if result['success']:
            print(f"✅ {package} simulation successful")
            print(f"  Output: {result['stdout'][:200]}...")
        else:
            print(f"❌ {package} simulation failed")
            print(f"  Error: {result['stderr'][:200]}...")
            analysis['installation_errors'].append({
                'package': package,
                'error': result['stderr']
            })
    
    return analysis

def analyze_system_dependencies():
    """Analyze system-level dependencies for OpenCV"""
    print_section("SYSTEM DEPENDENCIES ANALYSIS", 1)
    
    analysis = {
        'termux_packages': {},
        'system_libraries': {},
        'missing_dependencies': [],
        'pkg_manager_health': False,
        'issues': []
    }
    
    # Check pkg manager health
    print_section("Package Manager Health", 2)
    result = run_command_safe("pkg list-installed | head -5")
    if result['success']:
        analysis['pkg_manager_health'] = True
        print("✅ Termux package manager working")
    else:
        analysis['issues'].append("Termux package manager not working")
        print("❌ Termux package manager issues")
        return analysis
    
    # Check for OpenCV system package
    print_section("System OpenCV Packages", 2)
    opencv_system_packages = ['opencv', 'libopencv', 'opencv-dev']
    
    for package in opencv_system_packages:
        result = run_command_safe(f"pkg show {package}")
        if result['success']:
            analysis['termux_packages'][package] = 'available'
            print(f"✅ {package}: Available in Termux")
            
            # Check if installed
            result2 = run_command_safe(f"pkg list-installed | grep {package}")
            if result2['success'] and result2['stdout']:
                analysis['termux_packages'][package] = 'installed'
                print(f"  📦 {package}: Already installed")
        else:
            analysis['termux_packages'][package] = 'not_available'
            print(f"❌ {package}: Not available in Termux")
    
    # Check for essential build dependencies
    print_section("Build Dependencies", 2)
    essential_deps = [
        'python', 'python-dev', 'python-pip',
        'clang', 'make', 'cmake', 'ninja',
        'libjpeg-turbo', 'libpng', 'libtiff',
        'numpy', 'pkg-config'
    ]
    
    for dep in essential_deps:
        result = run_command_safe(f"pkg show {dep}")
        if result['success']:
            # Check if installed
            result2 = run_command_safe(f"pkg list-installed | grep ^{dep}")
            if result2['success'] and result2['stdout']:
                analysis['termux_packages'][dep] = 'installed'
                print(f"✅ {dep}: Installed")
            else:
                analysis['termux_packages'][dep] = 'available'
                analysis['missing_dependencies'].append(dep)
                print(f"⚠️  {dep}: Available but not installed")
        else:
            analysis['termux_packages'][dep] = 'not_available'
            analysis['missing_dependencies'].append(dep)
            print(f"❌ {dep}: Not available")
    
    # Check system libraries
    print_section("System Libraries", 2)
    important_libs = [
        'libc.so', 'libm.so', 'libz.so', 'libpng.so', 'libjpeg.so'
    ]
    
    for lib in important_libs:
        result = run_command_safe(f"find $PREFIX/lib -name '{lib}*' 2>/dev/null")
        if result['success'] and result['stdout']:
            analysis['system_libraries'][lib] = result['stdout'].strip().split('\n')
            print(f"✅ {lib}: Found")
            for path in analysis['system_libraries'][lib]:
                print(f"  📁 {path}")
        else:
            analysis['system_libraries'][lib] = None
            print(f"❌ {lib}: Not found")
    
    return analysis

def analyze_opencv_specific_issues():
    """Analyze OpenCV-specific compilation and runtime issues"""
    print_section("OPENCV-SPECIFIC ISSUES ANALYSIS", 1)
    
    analysis = {
        'compilation_flags': {},
        'known_issues': [],
        'workarounds': [],
        'performance_factors': []
    }
    
    # Check for known OpenCV compilation issues on ARM/Termux
    print_section("Known Issues Check", 2)
    
    known_issues = [
        {
            'name': 'NEON instruction set',
            'check': lambda: 'neon' in open('/proc/cpuinfo').read().lower() if os.path.exists('/proc/cpuinfo') else False,
            'impact': 'Performance optimization, may cause compilation issues',
            'workaround': 'Disable NEON: export CMAKE_ARGS="-DCPU_BASELINE=0"'
        },
        {
            'name': 'Memory limitations during compilation',
            'check': lambda: True,  # Always relevant
            'impact': 'Compilation may fail with out-of-memory errors',
            'workaround': 'Use swap file or reduce parallel compilation jobs'
        },
        {
            'name': 'Missing GUI libraries',
            'check': lambda: True,
            'impact': 'Full OpenCV package may fail due to missing X11/Qt',
            'workaround': 'Use opencv-python-headless instead'
        },
        {
            'name': 'ARM cross-compilation issues',
            'check': lambda: platform.machine().lower() in ['aarch64', 'arm64', 'armv7l'],
            'impact': 'Pre-compiled wheels may not be available',
            'workaround': 'Compile from source or use alternative packages'
        }
    ]
    
    for issue in known_issues:
        try:
            affects_system = issue['check']()
            if affects_system:
                analysis['known_issues'].append(issue)
                print(f"⚠️  {issue['name']}")
                print(f"   Impact: {issue['impact']}")
                print(f"   Workaround: {issue['workaround']}")
            else:
                print(f"✅ {issue['name']}: Not an issue on this system")
        except Exception as e:
            print(f"❓ {issue['name']}: Could not check ({e})")
    
    # Check compilation environment
    print_section("Compilation Environment", 2)
    
    # Check for conflicting environment variables
    problematic_vars = {
        'CFLAGS': 'May interfere with OpenCV compilation',
        'CXXFLAGS': 'May interfere with OpenCV compilation', 
        'LDFLAGS': 'May interfere with linking',
        'CMAKE_ARGS': 'May override OpenCV CMake settings'
    }
    
    for var, impact in problematic_vars.items():
        value = os.environ.get(var)
        if value:
            analysis['compilation_flags'][var] = value
            print(f"⚠️  {var}={value}")
            print(f"   {impact}")
        else:
            print(f"✅ {var}: Not set")
    
    return analysis

def generate_comprehensive_report(system_analysis, python_analysis, installation_analysis, deps_analysis, opencv_analysis):
    """Generate a comprehensive report with root cause analysis and recommendations"""
    print_section("COMPREHENSIVE ROOT CAUSE ANALYSIS REPORT", 1)
    
    all_issues = []
    all_issues.extend(system_analysis.get('issues', []))
    all_issues.extend(python_analysis.get('issues', []))
    all_issues.extend(installation_analysis.get('issues', []))
    all_issues.extend(deps_analysis.get('issues', []))
    
    # Categorize issues by severity
    critical_issues = []
    warning_issues = []
    info_issues = []
    
    for issue in all_issues:
        if any(keyword in issue.lower() for keyword in ['critical', 'no c compiler', 'cmake', 'not running in termux']):
            critical_issues.append(issue)
        elif any(keyword in issue.lower() for keyword in ['memory', 'storage', 'wheel']):
            warning_issues.append(issue)
        else:
            info_issues.append(issue)
    
    print_section("CRITICAL ISSUES (Must Fix)", 2)
    if critical_issues:
        for issue in critical_issues:
            print(f"🚨 {issue}")
    else:
        print("✅ No critical issues found")
    
    print_section("WARNING ISSUES (Should Fix)", 2)
    if warning_issues:
        for issue in warning_issues:
            print(f"⚠️  {issue}")
    else:
        print("✅ No warning issues found")
    
    print_section("INFORMATIONAL ISSUES", 2)
    if info_issues:
        for issue in info_issues:
            print(f"ℹ️  {issue}")
    else:
        print("✅ No informational issues found")
    
    # Generate recommendations
    print_section("RECOMMENDED SOLUTIONS (Priority Order)", 1)
    
    recommendations = []
    
    # Check if we have working alternatives
    if python_analysis['installed_packages'].get('numpy'):
        recommendations.append({
            'priority': 1,
            'solution': 'Use Enhanced Compatibility Layer',
            'description': 'Your bot already has cv2_compat.py which provides all needed OpenCV functions',
            'command': 'python test_opencv.py  # Test if compatibility layer works',
            'why': 'Fastest solution, no compilation needed, works immediately'
        })
    
    # System package installation
    if deps_analysis.get('pkg_manager_health') and deps_analysis['termux_packages'].get('opencv') == 'available':
        recommendations.append({
            'priority': 2,
            'solution': 'Install System OpenCV Package',
            'description': 'Use pre-compiled OpenCV from Termux repositories',
            'command': 'pkg install opencv python',
            'why': 'Pre-compiled, faster than pip installation, likely to work'
        })
    
    # Headless installation if tools available
    if python_analysis['compilation_tools'].get('clang') and not critical_issues:
        recommendations.append({
            'priority': 3,
            'solution': 'Install OpenCV Headless',
            'description': 'Install minimal OpenCV without GUI dependencies',
            'command': 'pip install --no-cache-dir opencv-python-headless',
            'why': 'Lighter package, fewer dependencies, higher success rate'
        })
    
    # Fix missing dependencies
    if deps_analysis['missing_dependencies']:
        recommendations.append({
            'priority': 4,
            'solution': 'Install Missing Dependencies',
            'description': f"Install missing build tools: {', '.join(deps_analysis['missing_dependencies'][:5])}",
            'command': f"pkg install {' '.join(deps_analysis['missing_dependencies'][:10])}",
            'why': 'Required for any OpenCV compilation'
        })
    
    # Memory optimization
    if system_analysis.get('available_memory') and system_analysis['available_memory'] < 1024:
        recommendations.append({
            'priority': 5,
            'solution': 'Create Swap File',
            'description': 'Add virtual memory for compilation',
            'command': '''
# Create 1GB swap file
dd if=/dev/zero of=$PREFIX/swapfile bs=1M count=1024
chmod 600 $PREFIX/swapfile
mkswap $PREFIX/swapfile
swapon $PREFIX/swapfile''',
            'why': 'OpenCV compilation needs more memory than available'
        })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['solution']} (Priority {rec['priority']})")
        print(f"   📝 {rec['description']}")
        print(f"   💡 Why: {rec['why']}")
        print(f"   🔧 Command: {rec['command']}")
    
    # Final summary
    print_section("ROOT CAUSE SUMMARY", 1)
    
    if critical_issues:
        print("🚨 PRIMARY ROOT CAUSE: Critical system issues")
        print("   Most likely cause: Missing compilation tools or incompatible environment")
        print("   Solution: Fix critical issues first, then retry installation")
    elif len(warning_issues) >= 3:
        print("⚠️  PRIMARY ROOT CAUSE: Multiple system limitations")
        print("   Most likely cause: Resource constraints (memory/storage) or missing dependencies")
        print("   Solution: Use pre-compiled packages or compatibility layer")
    elif opencv_analysis['known_issues']:
        print("🔧 PRIMARY ROOT CAUSE: OpenCV-specific compilation issues")
        print("   Most likely cause: ARM architecture or Termux environment compatibility")
        print("   Solution: Use workarounds or alternative packages")
    else:
        print("✅ NO MAJOR ROOT CAUSE IDENTIFIED")
        print("   Your system appears capable of running OpenCV")
        print("   Try the recommended solutions in order")
    
    return {
        'critical_issues': critical_issues,
        'warning_issues': warning_issues,
        'recommendations': recommendations,
        'system_capable': len(critical_issues) == 0
    }

def main():
    """Main analysis function"""
    print("🔬 OpenCV Installation Root Cause Analyzer")
    print("=" * 60)
    print("This tool will perform deep analysis of your Termux environment")
    print("to identify why OpenCV installation fails and provide solutions.")
    print("=" * 60)
    
    # Perform all analyses
    system_analysis = analyze_system_environment()
    python_analysis = analyze_python_environment()
    installation_analysis = analyze_opencv_installation_attempts()
    deps_analysis = analyze_system_dependencies()
    opencv_analysis = analyze_opencv_specific_issues()
    
    # Generate comprehensive report
    report = generate_comprehensive_report(
        system_analysis, python_analysis, installation_analysis, 
        deps_analysis, opencv_analysis
    )
    
    # Save detailed analysis to file
    full_analysis = {
        'system': system_analysis,
        'python': python_analysis,
        'installation': installation_analysis,
        'dependencies': deps_analysis,
        'opencv': opencv_analysis,
        'report': report
    }
    
    try:
        with open('opencv_analysis_report.json', 'w') as f:
            json.dump(full_analysis, f, indent=2, default=str)
        print(f"\n📄 Detailed analysis saved to: opencv_analysis_report.json")
    except Exception as e:
        print(f"\n❌ Could not save detailed report: {e}")
    
    print(f"\n🎯 NEXT STEPS:")
    if report['system_capable']:
        print("1. Try the recommended solutions in priority order")
        print("2. Test each solution with: python test_opencv.py")
        print("3. If all fail, use the compatibility layer (cv2_compat.py)")
    else:
        print("1. Fix critical issues first")
        print("2. Consider using the compatibility layer as immediate solution")
        print("3. Retry analysis after fixing critical issues")
    
    return report['system_capable']

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n🚨 Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)