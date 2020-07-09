#!/bin/bash

sudo docker stop ubermejo-covid19-converter
sudo docker rm ubermejo-covid19-converter
sudo docker run --name ubermejo-covid19-converter -v covid19-model:/results ubermejo/converter-anaconda3
