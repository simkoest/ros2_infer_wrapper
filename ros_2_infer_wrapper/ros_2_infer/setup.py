from setuptools import setup
import os
from glob import glob

package_name = 'ros_2_infer'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.onnx'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='agrigaia',
    maintainer_email='matthias.igelbrink@hs-osnabrueck.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros_2_infer = ros_2_infer.ros_2_infer_node:main'
        ],
    },
)
