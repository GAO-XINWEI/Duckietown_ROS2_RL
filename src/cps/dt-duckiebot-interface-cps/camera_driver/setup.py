import os

from setuptools import setup
from glob import glob


package_name = 'camera_driver'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Andrea Censi',
    author_email='acensi@idsc.mavt.ethz.ch',
    maintainer='nicholas-gs',
    maintainer_email='nicholasganshyan@gmail.com',
    description='Package for interfacing with camera',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'jetson_nano_camera_node = camera_driver.jetson_nano_camera_node:main'
        ],
    },
)
