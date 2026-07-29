
# Overview
To run the video demo, you need to use host display inside the docker container. Follow this guide to setup.

# Assumption
Assume that you are using Ubuntu wayland based host OS. (Default configuration of Ubuntu)
Also it is highly suggested to work directly on the host PC not via ssh.

# Docker Run

To pass the display to the container, you have to map /tmp/.X11-unix with run option. See example below:

```bash
docker run -dit --device=/dev/bos/0:/dev/bos/0 \
  --cap-add=ALL \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -v /dev/mem:/dev/mem \
  -v /sys/bus/pci/devices/:/sys/bus/pci/devices/ \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
  --ipc=host \
  --name bos_host \
  bos_host:latest
```

# Allow Display access to docker container

You need to login to GUI on Host machine and check $DISPLAY variable. Usually it is :0 as below:
```bash
$ echo $DISPLAY
:0
```

Then allow access to the display from docker:
```bash
$ xhost +local:docker
```

After that set the DISPLAY variable same as host inside the docker
```bash
(In side the docker container)
$ export DISPLAY=:0
```
# Install Dependencies (inside docker)

Following packages need to be installed inside the docker container
```bash
apt-get update && apt-get install -y \  
    libx11-6 \  
    libxcb1 \  
    libx11-xcb1 \  
    libxcb-render0 \  
    libxcb-shape0 \  
    libxcb-shm0 \  
    libglu1-mesa \  
    libgl1-mesa-glx \  
    libqt5gui5 \  
    libqt5widgets5 \  
    libqt5core5a \  
    libqt5x11extras5
```


# Simply check the display

Inside docker, run simple X11 based app like xclock and check if it is displayed at host PC's monitor.

```bash
$ apt install x11-apps
$ xclock
```
