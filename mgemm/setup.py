from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys

class CMakeBuild(build_ext):
    def run(self):
        # 确保build目录存在
        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
        os.makedirs(build_dir, exist_ok=True)
        
        # 运行cmake配置
        cmake_cmd = [
            'cmake', 
            '-B', 'build', 
            '-S', '.'
        ]
        subprocess.check_call(cmake_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # 运行cmake构建
        build_cmd = [
            'cmake', 
            '--build', 'build', 
            '-j', '32'
        ]
        subprocess.check_call(build_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # 复制构建产物到扩展目录
        for ext in self.extensions:
            output = self.get_ext_fullpath(ext.name)
            os.makedirs(os.path.dirname(output), exist_ok=True)
            # 假设生成的库名为mixedgemm.so (Linux)或mixedgemm.dll (Windows)
            if sys.platform.startswith('win'):
                lib_path = os.path.join('build', 'Release', 'mixedgemm.dll')
            else:
                lib_path = os.path.join('build', 'mixedgemm.so')
            if os.path.exists(lib_path):
                self.copy_file(lib_path, output)

setup(
    name='mixedgemm',
    version='0.1',
    description='Mixed precision GEMM library',
    ext_modules=[
        Extension('mixedgemm', sources=[])  # 源文件通过CMake管理
    ],
    cmdclass={
        'build_ext': CMakeBuild,
    },
    install_requires=[
        'torch>=2.0.0',
        'pybind11>=2.10.0',
    ],
)