import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tugbot_recorder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        #(os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='newin',
    maintainer_email='newinjollyk@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'recorder = tugbot_recorder.recorder:main',
            'odm_log = tugbot_recorder.odm_log:main',
            'lnn_prediction_01 = tugbot_recorder.lnn_prediction_01:main',
            'safety_node = tugbot_recorder.safety_node:main',
        ],
    },
)
