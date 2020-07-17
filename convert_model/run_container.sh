#!/bin/bash

sudo docker stop ubermejo-covid19-converter
sudo docker rm ubermejo-covid19-converter
sudo docker run -it --ipc=host --gpus "device=0" --name ubermejo-covid19-converter -v covid19-model:/results ubermejo/converter-anaconda3 bin/bash
