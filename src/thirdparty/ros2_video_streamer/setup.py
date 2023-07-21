from setuptools import setup, find_packages
import os
from glob import glob

PACKAGE_NAME = 'ros2_video_streamer'

setup(
    name=PACKAGE_NAME,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + PACKAGE_NAME]),
        ('share/' + PACKAGE_NAME, ['package.xml']),
        (os.path.join('share', PACKAGE_NAME, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', PACKAGE_NAME, 'config'), glob('config/*.yaml')),
    ],
    py_modules=[],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'natsort'
    ],
    author='Andreas Klintberg',
    maintainer='Andreas Klintberg',
    keywords=['ROS2'],
    description='Camera simulator - run a recorded video file streamed on a ros topic as if a live webcamera.',
    license='Apache License, Version 2.0',
    test_suite='test',
    entry_points={
        'console_scripts': [
            'ros2_video_streamer_node = ros2_video_streamer.ros2_video_streamer_node:main',
        ],
    },
)
