#!/bin/bash
set -e

ROS_DIST_SETUP_FILE=/opt/ros/humble/setup.bash
ROS_WS_SETUP_FILE=~/ros2_ws/install/setup.bash

source "$ROS_DIST_SETUP_FILE"
source "$ROS_WS_SETUP_FILE"

echo "#############"
echo "Sourced '$ROS_DIST_SETUP_FILE' and '$ROS_WS_SETUP_FILE'"
echo "Executing '$@'"
echo "#############"

exec "$@"
