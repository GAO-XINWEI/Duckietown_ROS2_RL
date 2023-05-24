import os

from setuptools import setup
from glob import glob

package_name = 'stop_line_filter'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nicholas-gs',
    maintainer_email='nicholasganshyan@gmail.com',
    description='Detect red stop line from detected line segments',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stop_line_filter_node=stop_line_filter.stop_line_filter_node:main',
            'stop_line_filter_tester_node=stop_line_filter.stop_line_filter_tester_node:main'
        ],
    },
)
