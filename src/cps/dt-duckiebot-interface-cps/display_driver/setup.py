import os

from setuptools import setup
from glob import glob

package_name = 'display_driver'

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
    author='Andrea F. Daniele',
    author_email='afdaniele@ttic.edu',
    maintainer='Nicholas Gan',
    maintainer_email='nicholasganshyan@gmail.com',
    description='An interface to the LCD display',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'display_driver_node = display_driver.display_driver_node:main'
        ],
    },
)
