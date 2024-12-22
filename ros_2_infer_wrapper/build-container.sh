#!/bin/bash

DOCKER_BUILDKIT=1 docker build --no-cache=true -t inference_node_test .
