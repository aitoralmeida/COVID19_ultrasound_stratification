#!/bin/bash

sudo docker stop ubermejo-covid19
sudo docker rm ubermejo-covid19
sudo nvidia-docker run --name ubermejo-covid19 -v covid19-model:/results ubermejo/nvidia-cuda 
