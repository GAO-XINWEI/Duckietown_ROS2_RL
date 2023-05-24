import os

from setuptools import setup
from glob import glob

package_name = 'rl_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xinwei',
    maintainer_email='1362509665@qq.com',
    description='RL Control Node: Input image, output action.',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'rl_control_node = rl_control_node.rl_control_node:main'
        ],
    },
)
