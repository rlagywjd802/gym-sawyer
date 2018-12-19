.PHONY: build-garage-headless-ros run-garage-headless-ros \
	build-garage-nvidia-ros run-garage-nvidia-ros \
	build-nvidia-sawyer-sim run-nvidia-sawyer-sim \
	build-nvidia-sawyer-robot run-nvidia-sawyer-robot

GARAGE_TYPE ?= headless
GARAGE_VER ?= latest
export GARAGE_TYPE
export GARAGE_VER

# Sets the add-host argument used to connect to the Sawyer ROS master
SAWYER_NET = "$(SAWYER_HOSTNAME):$(SAWYER_IP)"
ifneq (":", $(SAWYER_NET))
	ADD_HOST=--add-host=$(SAWYER_NET)
endif

build-garage-headless-ros: docker/docker-compose-garage-ros-intera.yml
	docker-compose -f docker/docker-compose-garage-ros-intera.yml build

build-garage-nvidia-ros: GARAGE_TYPE=nvidia
build-garage-nvidia-ros: docker/docker-compose-garage-ros-intera.yml
	docker-compose -f docker/docker-compose-garage-ros-intera.yml build

build-nvidia-sawyer-sim: docker/docker-compose-nv-sim.yml docker/get_intera.sh
	docker/get_intera.sh --sim
	docker-compose -f docker/docker-compose-nv-sim.yml build

build-nvidia-sawyer-robot: docker/docker-compose-nv-robot.yml docker/get_intera.sh
	docker/get_intera.sh
	docker-compose -f docker/docker-compose-nv-robot.yml build

run-garage-headless-ros: build-garage-headless-ros
	@ docker run \
		--init \
		-it \
		--rm \
		--net="host" \
		$(ADD_HOST) \
		-e MJKEY="$(MJKEY)" \
		--name "garage-headless-ros" \
		rlworkgroup/garage-headless-ros-intera:$(GARAGE_VER) $(RUN_CMD)

run-garage-nvidia-ros: build-garage-nvidia-ros
	xhost +local:docker
	@ docker run \
		--init \
		-it \
		--rm \
		--runtime=nvidia \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY="${DISPLAY}" \
		-e QT_X11_NO_MITSHM=1 \
		--net="host" \
		$(ADD_HOST) \
		-e MJKEY="$(MJKEY)" \
		--name "garage-nvidia-ros" \
		rlworkgroup/garage-nvidia-ros-intera:$(GARAGE_VER) $(RUN_CMD)

run-nvidia-sawyer-sim: build-nvidia-sawyer-sim
	xhost +local:docker
	docker run \
		--init \
		-t \
		--rm \
		--runtime=nvidia \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY="${DISPLAY}" \
		-e QT_X11_NO_MITSHM=1 \
		--net="host" \
		--name "sawyer-sim" \
		gym-sawyer/nvidia-sawyer-sim

run-nvidia-sawyer-robot: build-nvidia-sawyer-robot
ifeq (,$(ADD_HOST))
	$(error Set the environment variables SAWYER_HOST and SAWYER_IP)
endif
	xhost +local:docker
	docker run \
		--init \
		-t \
		--rm \
		--runtime=nvidia \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY="${DISPLAY}" \
		-e QT_X11_NO_MITSHM=1 \
		--net="host" \
		$(ADD_HOST) \
		--name "sawyer-robot" \
		gym-sawyer/nvidia-sawyer-robot
