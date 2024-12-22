#FROM nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime
FROM ubuntu:22.04

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Add ROS repository
RUN apt update && apt install -y    curl gnupg2 lsb-release && \
        curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
        sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# install dependencies and run upgrade
RUN apt update && \
    apt install -y --no-install-recommends unattended-upgrades && \
    unattended-upgrade

RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    libopenblas-dev \
    libpython3-dev \
    #libpython3.8-dev \
    python3-pip \
    python3-dev \
    cmake \
    wget \
    ros-humble-ros-base \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-sensor-msgs \
    ros-humble-sensor-msgs-py

# install basic python dependencies
RUN pip3 install --upgrade pip && \
        pip3 install setuptools && \
        pip3 install wheel pybind11 pytest

WORKDIR /root

# get requirements.txt file and the onnxruntime binary
RUN wget https://nvidia.box.com/shared/static/v59xkrnvederwewo2f1jtv6yurl92xso.whl -O onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
COPY ros_2_infer/requirements.txt .

# install python dependencies
RUN pip3 install --ignore-installed -r requirements.txt
RUN pip3 install -U colcon-common-extensions

# build ros node
RUN mkdir -p ros2_ws/src/
COPY ros_2_infer ros2_ws/src/ros_2_infer
RUN bash -c "cd ros2_ws && source /opt/ros/humble/setup.bash && colcon build"

#COPY models /models

# setup container entrypoint
COPY docker-entrypoint.sh .
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["ros2", "run", "ros_2_infer", "ros_2_infer"]
