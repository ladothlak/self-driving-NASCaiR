# DaiLE

A project where we are trying to build a model to drive cars on famous ovals/speedways! Features a CNN-LSTM network implemented in PyTorch. Currently, it uses telemetry and track footage to determine the optimal inputs.

We originally set it up to use WASD for the inputs, but after extensive testing have decided to explore controller input instead. This transformed the driving from a classification task (strictly right/wrong answers) to a regression problem. This should allow for more interesting emergent behavior and improved performance overall.

*Note:* Getting the controller branch of DaiLE can be somewhat irksome (as you'll see from the prerequisites). If you want a quick test drive, I recommend checking out the master branch instead.

## Prerequisites

To use this branch of DaiLE, in addition to the libraries used within the project, you will need to download and install [vjoy drivers](http://vjoystick.sourceforge.net/site/index.php/download-a-install/download) and [pyvjoy](https://github.com/tidzo/pyvjoy). DaiLE uses these to input controller commands.

Additionally, you will need to have either Steam (preferred) or x360ce installed on your computer, and you'll have to map the DaiLE outputs to the controllers. I have noticed that vjoy and x360ce sometimes do not interact well with one another, so I recommend using Steam's controller mappings.

## What's included with this package

DaiLE.py: a class that loads the trained model and can interact with the game

record_data.py: a class that allows for asyncronous data acquisition of telemetry, visual game state (i.e. screenshots), and controller inputs

play_together.py: a script which allows for cross-play between you and DaiLE--good for getting him out sticky situations. Actuating either the throttle or brakes will immediately interrupt DaiLE's game time and let you help him out

train.py: script to train a new DaiLE model

test_rnn.py: the actual architecture behind DaiLE's model. Still subject to change with time, as we are investigating features to give him still


