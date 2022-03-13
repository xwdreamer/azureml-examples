# Install Pre-requisites

* Install WSL2 on windows 11
https://docs.microsoft.com/en-us/windows/wsl/install

* Install docker desktop
https://hub.docker.com/editions/community/docker-ce-desktop-windows

Verify docker desktop is running with "Linux container" mode

![Switch to this](linux_container_mode.jpg "docker desktop properties")

Open command prompt/powershell

Get the full content of this [dockerbuild folder](dockerbuild)

Go inside this folder and run docker build on [this docker file](dockerbuild\sdk_test_ubuntu.dockerfile)
```
cd dockerbuild
docker build --pull --rm -f "sdk_test_ubuntu.dockerfile" -t sdkv2-samples:latest "." 
```

Start the docker container
```
docker run --name sample_container --rm -p 8888:8888 -it sdkv2-samples:latest
```

Run this command in the container:

```
bash .start_jupyter.sh
```