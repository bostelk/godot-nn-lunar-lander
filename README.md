# godot-nn-lunar-lander
An example Godot project to solve the [Lunar Lander Gym](https://gymnasium.farama.org/environments/box2d/lunar_lander/) using a nerual network.

The neural network is loaded and run in a compute shader. The shaders are written in [Slang](https://shader-slang.org) to improve portability.

The neural network behavoiur is not invariant between gyms because the state space it not similar, and would have to be realigned.