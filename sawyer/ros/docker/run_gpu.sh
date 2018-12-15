#!/bin/bash

SAWYER_HOSTNAME="021707CP00056.local"
SAWYER_IP="192.168.33.7"
ENTRYPOINT_DIR="/root/code/gym-sawyer/sawyer/ros/docker/entrypoints"
ENTRYPOINT="${ENTRYPOINT_DIR}/entrypoint-empty.sh"
#ENTRYPOINT="/bin/bash"

if [ -n "${1}" ]; then
	if [[ "${1}" == "sim" ]]; then
		ENTRYPOINT="${ENTRYPOINT_DIR}/entrypoint-sim.sh"
	elif  [[ "${1}" == "april" ]]; then
		ENTRYPOINT="${ENTRYPOINT_DIR}/entrypoint-robot-april.sh"
	elif [[ "${1}" == "robot" ]]; then
		ENTRYPOINT="${ENTRYPOINT_DIR}/entrypoint-robot.sh"
	fi
fi

xhost +local:root

if [ -z ${NVIDIA_DRIVER+x} ]; then
	NVIDIA_DRIVER=$(nvidia-settings -q NvidiaDriverVersion | head -2 | tail -1 | sed 's/.*\([0-9][0-9][0-9]\)\..*/\1/') ;
fi
if [ -z ${NVIDIA_DRIVER+x} ]; then
	echo "Error: Could not determine NVIDIA driver version number. Please specify your driver version number manually in $0." 1>&2 ;
	exit ;
else
	echo "Linking to NVIDIA driver version $NVIDIA_DRIVER..." ;
fi

DOCKER_VISUAL_NVIDIA="-v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/nvidia0 --device /dev/nvidiactl"

docker run \
	-it \
	--rm \
	--runtime=nvidia \
	--init \
	--privileged \
	$DOCKER_VISUAL_NVIDIA \
	--net="host" \
	--add-host="${SAWYER_HOSTNAME}:${SAWYER_IP}" \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--cap-add SYS_ADMIN \
	--cap-add MKNOD \
	--device /dev/fuse:/dev/video0 \
	--volume=/dev/bus/usb:/dev/bus/usb:ro \
	--name "sawyer-ros-docker" \
	--security-opt apparmor:unconfined \
	--entrypoint ${ENTRYPOINT} \
sawyer-ros-docker:gpu bash;

xhost -local:root
