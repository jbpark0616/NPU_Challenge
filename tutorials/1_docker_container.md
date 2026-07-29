# Docker Container

## Docker Overview

### 이미지와 컨테이너

Docker는 독립적인 가상 컨테이너를 제공하여 소프트웨어가 실행되는 환경을 호스트와 격리할 수 있도록 하는 소프트웨어이다. Docker의 컨테이너는 특정한 이미지를 기반으로 만들어지며, 다양한 베이스 이미지를 통해 소프트웨어가 실행되는 환경을 설정할 수 있다. 이미지 생성은 일반적으로 1) Docker Hub에 업로드된 이미지를 활용하거나 2) Dockerfile을 활용하여 커스텀 이미지를 구성하는 방식으로 이루어진다.

### 주요 명령어 및 사용법

| Command | Description |
| :--- | :--- |
| `docker ps` | 현재 실행 중인 docker 컨테이너를 확인한다 |
| `docker ps -a` | 현재 실행 중이거나 종료된 docker 컨테이너를 확인한다 |
| `docker images` | 현재 docker에 저장된 이미지를 확인한다 |
| `docker create <options> --name <container> <image>` | docker 컨태이너를 생성한다 |
| `docker start <container name>` | 해당 docker 컨테이너를 실행한다 |
| `docker restart <container name>` | 해당 docker 컨테이너를 재실행한다 |
| `docker run <options> --name <container> <image>` |  docker 컨테이너를 생성하고 실행한다 (create + start) |
| `docker stop <container>` | docker 컨테이너를 종료한다 |
| `docker rm <container>` | docker 컨테이너를 삭제한다 |


## BOS Eagle N SDK Docker Container

### STEP 1: Docker 컨테이너 생성하기

주어진 자동화 스크립트를 활용하여 BOS SDK를 위한 컨테이너를 생성할 수 있다. 예를 들어 이미지가 `bos_host:fc_rc5.1`이고 컨테이너 이름이 `bos_host`인 경우 다음과 같이 실행한다.

```bash
./autorun_docker.sh -i bos_host:fc_rc5.1 -c bos_host
```

이후 `docker ps` 명령어를 통해 컨테이너가 정상적으로 생성되고 실행되는 중인지 확인한다.

> 현재 서버에 설치된 BOS SDK의 최신 docker 이미지는 `bos_host:fc_rc5.1` 이다.

> 포트포워딩으로 인해 여러 개의 컨테이너를 동시에 실행하는 것은 불가능하다. 만약 여러 개의 컨테이너를 생성하고자 한다면, `docker stop` 명령어를 통해 현재 실행중인 컨테이너를 중지하고 다른 컨테이너를 생성한다. 이전에 종료한 컨테이너를 재실행하고자 한다면 `docker restart` 명령어를 활용한다.

> 서버를 재부팅하면 기존에 실행되던 docker 컨테이너들은 자동으로 종료된다.

#### Optional: `autorun_docker.sh`가 정상적으로 실행되지 않는 경우

아래 커맨드를 통해 docker 컨테이너를 생성한다.

```bash
export IMG_NAME="bos_host:fc_rc5.1"  # latest BOS SDK image
export CONTAINER="bos_host"

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
```

`docker run`에 사용된 주요 옵션들은 다음과 같다.

| Option | Description |
| :--- | :--- |
| `--device=/dev/bos/0:/dev/bos/0` | BOS Eagle N 보드를 docker 컨테이너에서 사용하기 위한 옵션 |
| `-v <host volume>:<container volume>` | 호스트의 볼륨을 docker 컨테이너에서 마운트하기 위한 옵션으로 여기서는 hugepage 관련 볼륨과 PCIe 장치 관련 볼륨을 마운트 |
| `-p <host port>:<container port>` | 호스트의 포트를 docker 컨테이너의 포트로 포워딩하는 옵션으로 Tracy 및 TTNN 출력을 위해 사용 |
| `-e DISPLAY=$DISPLAY` | 호스트의 디스플레이 서버와 docker 컨테이너의 디스플레이 서버를 일치시키기 위해 docker 컨테이너에서 `DISPLAY` 환경 변수를 지정 |


> `xhost +local:docker` 명령어를 docker 컨테이너 생성 이전에 실행하여 X11 호스팅을 위한 docker 권한을 제공해야 `qt` 관련 화면 출력이 가능하다. 이는 BOS에서 제공하는 데모 실행 및 OpenCV 관련 작업에 필요할 수 있으니 참고할 것.

### STEP 2: Docker 컨테이너에서 bash shell 실행하기

아래 커맨드를 통해 docker 컨테이너에서 bash shell을 실행한다.

```bash
docker exec -it bos_host /bin/bash
```

실행 이후 아래와 같이 shell에 접속한 것을 확인할 수 있다.

```
root@<container ID>:/tt-metal# 
```

### STEP 3: 데모 실행하기

```bash
source env_set.sh   # 가상 환경 활성화
python models/bos_model/resnet50/run_resnet50.py    # ResNet50 데모 실행
```


## VSCode를 활용한 docker 컨테이너 접속

VSCode를 활용하여 docker 컨테이너에 접속할 수 있다. VSCode Extensions 메뉴에서 `Remote Explorer` 및 `Dev Containers` 확장을 설치 후 Remote Explorer 메뉴에서 상단 드롭다운 메뉴를 Dev Containers로 변경한다. 해당 메뉴에서 현재 실행중인 docker 컨테이너를 확인할 수 있다. 이후 Remote SSH 접속과 동일한 방식으로 컨테이너에 접속한다.