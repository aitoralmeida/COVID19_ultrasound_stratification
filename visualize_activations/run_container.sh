#!/bin/bash

sudo docker stop ubermejo-covid19-visualize
sudo docker rm ubermejo-covid19-visualize
sudo docker run -it --ipc=host --gpus "device=0" --name ubermejo-covid19-visualize -v covid19-model:/results ubermejo/covid-19-visualize bin/bash
