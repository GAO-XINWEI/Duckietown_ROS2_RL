import os

from setuptools import setup
from glob import glob

package_name = 'imu_driver'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Timothy Scott',
    maintainer_email='scottti@student.ethz.ch',
    description='Interface for reading sensors on DBV2 and publishing ROS topics',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_node = imu_driver.imu_node:main'
        ],
    },
)
