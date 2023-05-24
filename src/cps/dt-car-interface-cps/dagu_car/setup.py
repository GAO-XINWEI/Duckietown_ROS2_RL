
import os

from setuptools import setup
from glob import glob

package_name = 'dagu_car'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, f'{package_name}.test'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/test/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mack',
    maintainer_email='mack@duckietown.org',
    description='dagu_car',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kinematics_node = dagu_car.kinematics_node:main',
            'velocity_to_pose_node = dagu_car.velocity_to_pose_node:main',
            'car_cmd_switch_node = dagu_car.car_cmd_switch_node:main',
            'gatekeeper_node = dagu_car.gatekeeper_node:main',
            'test_kinematics_node = dagu_car.test.test_kinematics_node:main',
            'test_vtp_node = dagu_car.test.test_vtp_node:main'
        ],
    },
)
