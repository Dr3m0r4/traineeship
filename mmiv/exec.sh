#!/bin/bash

if [ $1 = train ]
then
  net_run train -a decay_lr_application.DecayLearningRateApplication -c config.ini
elif [ $1 = inference ]
then
  net_segment inference -c config.ini
fi
