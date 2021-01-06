# DaiLE

A project where we are trying to build a model to drive cars on famous ovals/speedways! Features a CNN-LSTM network implemented in PyTorch. Currently, it uses telemetry and track footage to determine the optimal inputs.

We originally set it up to use WASD for the inputs, but after extensive testing have decided to explore controller input instead. This will transform the driving from a classification task (strictly right/wrong answers) to a regression problem. We believe this will allow for more interesting emergent behavior and improved performance overall.

