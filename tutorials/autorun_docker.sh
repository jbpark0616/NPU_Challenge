#!/bin/bash

IMG_NAME="bos_host:fc_rc5.1"
CONTAINER="bos_host"

print_usage() {
    echo "USage: $0 [-i|--img-name <image_name>] [-c|--container] <container_name>]"
    echo ""
    echo "Options:"
    echo "  -i, --img-name   The name of the image (default: bos_host:fc_rc5.1)"
    echo "  -c, --container  The name of the container (default: bos_host)"
    echo "  -h, --help       Print this message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--img-name)
            if [[ -n "$2" && "$2" != -* ]]; then
                IMG_NAME="$2"
                shift 2
            else
                echo "Error: $1 option requires a value." >&2
                print_usage
            fi
            ;;
        -c|--container)
            if [[ -n "$2" && "$2" != -* ]]; then
                CONTAINER="$2"
                shift 2
            else
                echo "Error: $1 option requires a vlaue." >&2
                print_usage
            fi
            ;;
        -h|--help)
            print_usage
            ;;
        *) 
            echo "Error: Invalid option -> $1" >&2
            print_usage
            ;;
    esac
done


xhost +local:docker

docker run -dit \
    --device=/dev/bos/0:/dev/bos/0 \
    --cap-add=ALL \
    --ipc=host \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v /dev/mem:/dev/mem \
    -v /sys/bus/pci/devices/:/sys/bus/pci/devices/ \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 8000:8000 \
    -p 8086:8086 \
    -p 8087:8087 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name $CONTAINER $IMG_NAME


