using Godot;
using System;

public partial class RenderNeuralNetLinearLayer : Node3D
{
	/**
import torch
import torch.nn as nn

m = nn.Linear(5, 10)
print("weight")
print(m.weight)
print("bias")
print(m.bias)
input = torch.randn(2, 5)
print("input")
print(input)
print("output")
output = m(input)
print(output)
print(output.size())
torch.Size([2, 10])

weight
Parameter containing:
tensor([[ 0.2355,  0.1626,  0.1223,  0.1657, -0.0158],
		[-0.0559, -0.0994, -0.3926,  0.0721, -0.4406],
		[-0.0304,  0.1478,  0.2222, -0.3018, -0.4309],
		[-0.3522,  0.0759,  0.1537,  0.1691, -0.3935],
		[ 0.4243, -0.4332, -0.4007,  0.0327, -0.2722],
		[-0.0140, -0.3614, -0.1569, -0.2640, -0.2157],
		[ 0.2580,  0.0400, -0.4454,  0.3633, -0.0331],
		[ 0.2515, -0.2100,  0.3214, -0.2635, -0.3052],
		[ 0.1065, -0.3465, -0.3643, -0.3482,  0.3299],
		[ 0.2158, -0.1396, -0.1932,  0.3828,  0.1575]], requires_grad=True)
bias
Parameter containing:
tensor([-0.4145, -0.3780, -0.3884,  0.1104, -0.4200, -0.1506,  0.3056,  0.1951,
		 0.2584,  0.1053], requires_grad=True)
input	
tensor([[-0.2100,  1.1309,  0.6100,  0.5772,  0.0221],
		[ 0.4934,  0.6755, -1.6387,  1.3295, -0.2466]])
output
tensor([[-0.1101, -0.6862, -0.2631,  0.4529, -1.2306, -0.8093,  0.2339, -0.0580,
		 -0.5718,  0.0087],
		[-0.1645,  0.3751, -0.9626,  0.0579,  0.2639, -0.4423,  1.6810, -0.6244,
		  0.1296,  0.9040]], grad_fn=<AddmmBackward0>)
torch.Size([2, 10])
	*/
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		const int inBatches = 2;
		const int inFeatureSize = 5;
		const int outFeatureSize = 10;

		var rd = RenderingServer.CreateLocalRenderingDevice();
		
		// Load GLSL shader
		var shaderFile = GD.Load<RDShaderFile>("res://NeuralNetLinearLayer/nn-linear.glsl");
		var shaderBytecode = shaderFile.GetSpirV();
		var shader = rd.ShaderCreateFromSpirV(shaderBytecode);
		
		// Prepare our data. We use floats in the shader, so we need 32 bit.
		float[][] input = [
			[-0.2100f,  1.1309f,  0.6100f,  0.5772f,  0.0221f],
			[ 0.4934f,  0.6755f, -1.6387f,  1.3295f, -0.2466f]];
		var inputBytes = new byte[input.Length * input[0].Length * sizeof(float)];
		BlockCopy2D(input, inputBytes);

		float[][] weights = [
		[ 0.2355f,  0.1626f,  0.1223f,  0.1657f, -0.0158f],
		[-0.0559f, -0.0994f, -0.3926f,  0.0721f, -0.4406f],
		[-0.0304f,  0.1478f,  0.2222f, -0.3018f, -0.4309f],
		[-0.3522f,  0.0759f,  0.1537f,  0.1691f, -0.3935f],
		[ 0.4243f, -0.4332f, -0.4007f,  0.0327f, -0.2722f],
		[-0.0140f, -0.3614f, -0.1569f, -0.2640f, -0.2157f],
		[ 0.2580f,  0.0400f, -0.4454f,  0.3633f, -0.0331f],
		[ 0.2515f, -0.2100f,  0.3214f, -0.2635f, -0.3052f],
		[ 0.1065f, -0.3465f, -0.3643f, -0.3482f,  0.3299f],
		[ 0.2158f, -0.1396f, -0.1932f,  0.3828f,  0.1575f]];
		
		var weightsBytes = new byte[weights.Length * weights[0].Length * sizeof(float)];
		BlockCopy2D(weights, weightsBytes);

		float[] bias = [-0.4145f, -0.3780f, -0.3884f,  0.1104f, -0.4200f, -0.1506f,  0.3056f,  0.1951f,
		 0.2584f,  0.1053f];
		var biasBytes = new byte[bias.Length * sizeof(float)];
		Buffer.BlockCopy(bias, 0, biasBytes, 0, biasBytes.Length);

		float[] output = new float[inBatches * outFeatureSize];
		var outputBytes = new byte[output.Length * sizeof(float)];
		Buffer.BlockCopy(output, 0, outputBytes, 0, outputBytes.Length);

		// Create a storage buffer that can hold our float values.
		var buffer0 = rd.StorageBufferCreate((uint)inputBytes.Length, inputBytes);
		var buffer1 = rd.StorageBufferCreate((uint)weightsBytes.Length, weightsBytes);
		var buffer2 = rd.StorageBufferCreate((uint)biasBytes.Length, biasBytes);
		var buffer3 = rd.StorageBufferCreate((uint)outputBytes.Length, outputBytes);

		// Create a uniform to assign the buffer to the rendering device
		var inputUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 0
		};
		var weightsUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 1
		};
		var biasUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 2
		};
		var outputUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 3
		};
		inputUniform.AddId(buffer0);
		weightsUniform.AddId(buffer1);
		biasUniform.AddId(buffer2);
		outputUniform.AddId(buffer3);

		var uniformSet = rd.UniformSetCreate([inputUniform, weightsUniform, biasUniform, outputUniform], shader, 0);
		
		// Create a compute pipeline
		var pipeline = rd.ComputePipelineCreate(shader);
		var computeList = rd.ComputeListBegin();
		rd.ComputeListBindComputePipeline(computeList, pipeline);
		rd.ComputeListBindUniformSet(computeList, uniformSet, 0);
		rd.ComputeListDispatch(computeList, xGroups: outFeatureSize, yGroups: inBatches, zGroups: 1);
		rd.ComputeListEnd();
		
		// Submit to GPU and wait for sync
		rd.Submit();
		rd.Sync();
		
		// Read back the data from the buffers
		outputBytes = rd.BufferGetData(buffer3);
		output = new float[inBatches * outFeatureSize];
		Buffer.BlockCopy(outputBytes, 0, output, 0, outputBytes.Length);

		GD.Print("Input: ");
		for(int i = 0; i < input.Length; i++) {
			GD.Print(" ", string.Join(", ", input[i]));
		}

		float[] expectedOutput = [
			-0.1101f, -0.6862f, -0.2631f,  0.4529f, -1.2306f, -0.8093f,  0.2339f, -0.0580f,
		 -0.5718f,  0.0087f,
			-0.1645f,  0.3751f, -0.9626f,  0.0579f,  0.2639f, -0.4423f,  1.6810f, -0.6244f,
		  0.1296f,  0.9040f];

		bool[] outcome = new bool[expectedOutput.Length];
		GD.Print("Output: ");
		GD.Print(" ", string.Join(", ", output));
		for(int i = 0; i < output.Length; i++) {
			outcome[i] = Math.Abs(expectedOutput[i] - output[i]) < 0.001;
		}
		GD.Print("Outcome: ");
		GD.Print(" ", string.Join(", ", outcome));

		rd.Free();
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

	private void BlockCopy2D(float[][] src, byte[] dst) {
		int bytesOffset = 0;
		for(int i = 0; i < src.Length; i++) {
			int numBytes = src[i].Length * sizeof(float);
			Buffer.BlockCopy(src[i], 0, dst, bytesOffset, numBytes);
			bytesOffset += numBytes;
		}
	}
}
