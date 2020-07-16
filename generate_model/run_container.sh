#!/bin/bash

sudo docker stop ubermejo-covid19
sudo docker rm ubermejo-covid19
sudo docker run -it --ipc=host --gpus "device=0" --name ubermejo-covid19 -v covid19-model:/results ubermejo/covid-19-generate-model bin/bash
