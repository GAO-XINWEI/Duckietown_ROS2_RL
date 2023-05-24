from setuptools import setup

package_name = 'dt_time_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='nicholas-gs',
    author_email='nicholasganshyan@gmail.com',
    maintainer='nicholas-gs',
    maintainer_email='nicholasganshyan@gmail.com',
    description='Wrapper around builtin_interfaces time to make operations easier',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
