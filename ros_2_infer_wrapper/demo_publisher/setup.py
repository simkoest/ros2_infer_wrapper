from setuptools import setup
from glob import glob
import os

package_name = 'py_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resources'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'resources'), glob('py_pubsub/*.png')),
        (os.path.join('share', package_name, 'resources'), glob('py_pubsub/*.bin')),
        (os.path.join('share', package_name, 'resources'), glob('py_pubsub/*.jpg'))

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='koesse',
    maintainer_email='simon.koesters@hs-osnabrueck.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = py_pubsub.publisher_member_function:main',
        ],
    },
)
