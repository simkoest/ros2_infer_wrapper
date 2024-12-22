#!/bin/bash

docker run -ti --net host -v /dev/shm:/dev/shm --rm --name=inference_client inference_node_test "$@"

#docker run -it --rm --privileged=true \
#    --name="inference_client" \
#    --network="host" \
#    -v "/dev:/dev" \
#    inference_node_test /bin/bash -c "source /opt/ros/foxy/setup.bash ; source ros2_ws/install/setup.bash; ros2 run ros_2_infer ros_2_infer"

#docker exec -it bash -u "export PYTHONPATH=$(realpath /root/grpc_src/generated):$PYTHONPATH && source /opt/ros/foxy/setup.bash && source ros2_ws/install/setup.bash  && source .bashrc && ros2 run ros_2_infer ros_2_infer"

