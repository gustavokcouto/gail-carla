#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
screen -L -S carla_exp .env/bin/python carla_exp.py