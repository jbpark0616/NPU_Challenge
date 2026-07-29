# **How to use Docker container**

Docker image and Dockerfile could be used instead of installing the software stack on the host PC. Docker image contains all the runtime enviroment for AI workloads or you can build your own Docker image using the provided Dockerfile. You can run the AI workload by either way so that you can choose one option.

Prior to use Docker container, the Hugepage and KMD should be configured in your host PC. Please refer the 'Configure Hugepage on Linux' and 'Installing KMD driver' sections from [How to Install SW Stack](./How_to_Install_SW_Stack.md) document.
<br>

## Download docker image

Please refer to the download link of Docker image from the release note and download the image. The downloaded file might be compressed using gzip or 7zip, so extract Docker image using gunzip or 7zip before loading in docker. 

- eg) unpack the downloaded file and load image in docker  
    ```bash
    $ gunzip bos_host.tar.gz
    $ docker load -i bos_host.tar
    ``` 
- Then you can see the docker image  
    ```
    $ docker images
    IMAGE                 ID             DISK USAGE   CONTENT SIZE   EXTRA
    bos_host:latest       d3ea56eb5d49       17.1GB             0B 
    ```
<br>

## Build your own Docker image

Clone the tt-metal repository on your host PC. Please refer the 'Installing `tt-metal`' section from the release note. You don't need to build `tt-metal` when you use Docker container.  

- Copy the `Dockerfile` to the `tt-metal` directory.
    ```
    $ cp docker/Dockerfile ${TT_METAL_DIRECTORY_ROOT}
    $ cd ${TT_METAL_DIRECTORY_ROOT}
    $ dockr build -t bos_host .
    ```
- Then you can see the docker image.
    ```
    $ docker images
    IMAGE                 ID             DISK USAGE   CONTENT SIZE   EXTRA
    bos_host:latest       d3ea56eb5d49       17.1GB             0B 
    ```

<br>

## Run Docker container

Run the Docker container using the Docker image  
```
$ docker run -dit --device=/dev/bos/0:/dev/bos/0 \
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

Attach the docker container
```
$ docker exec -it bos_host:latest /bin/bash
root@xxxxxxxxxxxxxx # 
```

Please refer the [Using Host Display inside docker](./Using_Host_Display_inside_Docker.md) when you need display for AI workload.

<br>

## Run AI models on the Eagle-N A0 board

To run an AI model, refer to the `README.md` file in each model directory specified in the release note.
