#!/bin/bash

if [ $1 = train ]
then
  cp ./decay_lr_application.py ~/.conda/envs/niftynet/lib/python3.7/site-packages/niftynet/contrib/learning_rate_schedule/
  net_run train -a niftynet.contrib.learning_rate_schedule.decay_lr_application.DecayLearningRateApplication -c config.ini
elif [ $1 = inference ]
then
  net_segment inference -c config.ini
fi
