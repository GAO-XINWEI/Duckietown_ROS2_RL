import os

from setuptools import setup
from glob import glob

package_name = 'tof_driver'

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
    maintainer='Andrea F. Daniele',
    maintainer_email='nicholasganshyan@gmail.com',
    description='afdaniele@ttic.edu',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tof_node = tof_driver.tof_node:main'
        ],
    },
)
