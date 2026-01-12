using Godot;
using System;

public partial class RenderNeuralNetLunarLander : Node
{
	const int inBatches = 1;
	const int inFeatureSize = 8;
	const int hiddenSize = 128;
	const int outFeatureSize = 4;
	const int relu = 255;

	RenderingDevice rd;
	Rid shader;

	Rid uniformSet;
	Rid uniformSet1;
	Rid uniformSet2;

	Rid buffer0; // input
	Rid buffer9; // output

	Rid[] storageBuffers;
	
	float[] output;
	byte[] outputBytes;

	/**
tensor([ 1.5035,  1.2645,  0.8818, -0.4202, -0.9009,  0.4167,  1.3025, -0.0522])
tensor([12.2133, 15.5770, 12.8879, 10.6199], grad_fn=<ViewBackward0>)
	*/
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		var nn = NeuralNetModelFile.ReadFile("Models/dqn_lunar_lander.bin");
		
		rd = RenderingServer.CreateLocalRenderingDevice();
		
		// Load GLSL shader
		var shaderFile = GD.Load<RDShaderFile>("res://NeuralNetLunarLander/nn-lunar-lander.glsl");
		var shaderBytecode = shaderFile.GetSpirV();
		shader = rd.ShaderCreateFromSpirV(shaderBytecode);
		
		// Prepare our data. We use floats in the shader, so we need 32 bit.
		float[] input = new float[8];
		var inputBytes = new byte[input.Length * sizeof(float)];
		Buffer.BlockCopy(input, 0, inputBytes, 0, inputBytes.Length);
		
		var layer1weightsBytes = new byte[nn.layers[0].weights.Length * sizeof(float)];
		Buffer.BlockCopy(nn.layers[0].weights, 0, layer1weightsBytes, 0, layer1weightsBytes.Length);

		var layer1biasBytes = new byte[nn.layers[0].bias.Length * sizeof(float)];
		Buffer.BlockCopy(nn.layers[0].bias, 0, layer1biasBytes, 0, layer1biasBytes.Length);

		var layer2weightsBytes = new byte[nn.layers[1].weights.Length * sizeof(float)];
		Buffer.BlockCopy(nn.layers[1].weights, 0, layer2weightsBytes, 0, layer2weightsBytes.Length);

		var layer2biasBytes = new byte[nn.layers[1].bias.Length * sizeof(float)];
		Buffer.BlockCopy(nn.layers[1].bias, 0, layer2biasBytes, 0, layer2biasBytes.Length);

		var layer3weightsBytes = new byte[nn.layers[2].weights.Length * sizeof(float)];
		Buffer.BlockCopy(nn.layers[2].weights, 0, layer3weightsBytes, 0, layer3weightsBytes.Length);

		var layer3biasBytes = new byte[nn.layers[2].bias.Length * sizeof(float)];
		Buffer.BlockCopy(nn.layers[2].bias, 0, layer3biasBytes, 0, layer3biasBytes.Length);

		output = new float[inBatches * hiddenSize];
		outputBytes = new byte[output.Length * sizeof(float)];
		Buffer.BlockCopy(output, 0, outputBytes, 0, outputBytes.Length);

		float[] output1 = new float[inBatches * hiddenSize];
		var output1Bytes = new byte[output1.Length * sizeof(float)];
		Buffer.BlockCopy(output1, 0, output1Bytes, 0, output1Bytes.Length);

		float[] output2 = new float[inBatches * outFeatureSize];
		var output2Bytes = new byte[output2.Length * sizeof(float)];
		Buffer.BlockCopy(output2, 0, output2Bytes, 0, output2Bytes.Length);

		var linearParams = new int[4] { inFeatureSize, hiddenSize, relu, 0 };
		var linearParamsBytes = new byte[linearParams.Length * sizeof(float)];
		Buffer.BlockCopy(linearParams, 0, linearParamsBytes, 0, linearParamsBytes.Length);

		var linear1Params = new int[4] { hiddenSize, hiddenSize, relu, 0 };
		var linear1ParamsBytes = new byte[linear1Params.Length * sizeof(float)];
		Buffer.BlockCopy(linear1Params, 0, linear1ParamsBytes, 0, linear1ParamsBytes.Length);

		var linear2Params = new int[4] { hiddenSize, outFeatureSize, 0, 0 };
		var linear2ParamsBytes = new byte[linear2Params.Length * sizeof(float)];
		Buffer.BlockCopy(linear2Params, 0, linear2ParamsBytes, 0, linear2ParamsBytes.Length);

		// Create a storage buffer that can hold our float values.
		buffer0 = rd.StorageBufferCreate((uint)inputBytes.Length, inputBytes);
		
		var buffer1 = rd.StorageBufferCreate((uint)layer1weightsBytes.Length, layer1weightsBytes);
		var buffer2 = rd.StorageBufferCreate((uint)layer1biasBytes.Length, layer1biasBytes);
		
		var buffer3 = rd.StorageBufferCreate((uint)layer2weightsBytes.Length, layer2weightsBytes);
		var buffer4 = rd.StorageBufferCreate((uint)layer2biasBytes.Length, layer2biasBytes);
		
		var buffer5 = rd.StorageBufferCreate((uint)layer3weightsBytes.Length, layer3weightsBytes);
		var buffer6 = rd.StorageBufferCreate((uint)layer3biasBytes.Length, layer3biasBytes);
		
		var buffer7 = rd.StorageBufferCreate((uint)outputBytes.Length, outputBytes);
		var buffer8 = rd.StorageBufferCreate((uint)output1Bytes.Length, output1Bytes);
		buffer9 = rd.StorageBufferCreate((uint)output2Bytes.Length, output2Bytes);

		var buffer10 = rd.UniformBufferCreate((uint)linearParamsBytes.Length, linearParamsBytes);
		var buffer11 = rd.UniformBufferCreate((uint)linear1ParamsBytes.Length, linear1ParamsBytes);
		var buffer12 = rd.UniformBufferCreate((uint)linear2ParamsBytes.Length, linear2ParamsBytes);

		storageBuffers = [buffer0,buffer1,buffer2,buffer3,buffer4,buffer5,buffer6,buffer7,buffer8,buffer9,buffer10,buffer11,buffer12];
		
		// Create a uniform to assign the buffer to the rendering device
		var linearParamsUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.UniformBuffer,
			Binding = 0
		};
		var linear1ParamsUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.UniformBuffer,
			Binding = 0
		};
		var linear2ParamsUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.UniformBuffer,
			Binding = 0
		};
		var inputUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 1
		};
		var inputOutputUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 1
		};
		var inputOutput1Uniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 1
		};
		var layer1weightsUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 2
		};
		var layer1biasUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 3
		};
		var layer2weightsUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 2
		};
		var layer2biasUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 3
		};
		var layer3weightsUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 2
		};
		var layer3biasUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 3
		};
		var outputUniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 4
		};
		var output1Uniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 4
		};
		var output2Uniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 4
		};

		linearParamsUniform.AddId(buffer10);
		linear1ParamsUniform.AddId(buffer11);
		linear2ParamsUniform.AddId(buffer12);

		inputUniform.AddId(buffer0);
		inputOutputUniform.AddId(buffer7);
		inputOutput1Uniform.AddId(buffer8);

		layer1weightsUniform.AddId(buffer1);
		layer1biasUniform.AddId(buffer2);
		layer2weightsUniform.AddId(buffer3);
		layer2biasUniform.AddId(buffer4);
		layer3weightsUniform.AddId(buffer5);
		layer3biasUniform.AddId(buffer6);

		outputUniform.AddId(buffer7);
		output1Uniform.AddId(buffer8);
		output2Uniform.AddId(buffer9);

		uniformSet = rd.UniformSetCreate([
			linearParamsUniform,
			inputUniform,
			layer1weightsUniform,
			layer1biasUniform,
			outputUniform], shader, 0);
		
		uniformSet1 = rd.UniformSetCreate([
			linear1ParamsUniform,
			inputOutputUniform,
			layer2weightsUniform,
			layer2biasUniform,
			output1Uniform], shader, 0);

		uniformSet2 = rd.UniformSetCreate([
			linear2ParamsUniform,
			inputOutput1Uniform,
			layer3weightsUniform,
			layer3biasUniform,
			output2Uniform], shader, 0);
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{

	}

	public float[] PredictSync(float[] input)
	{
		var inputBytes = new byte[input.Length * sizeof(float)];
		Buffer.BlockCopy(input, 0, inputBytes, 0, inputBytes.Length);
		rd.BufferUpdate(buffer0, 0, (uint)inputBytes.Length, inputBytes);

		// Create a compute pipeline
		var pipeline = rd.ComputePipelineCreate(shader);

		var computeList = rd.ComputeListBegin();
		rd.ComputeListBindComputePipeline(computeList, pipeline);
		rd.ComputeListBindUniformSet(computeList, uniformSet, 0);
		rd.ComputeListDispatch(computeList, xGroups: hiddenSize, yGroups: inBatches, zGroups: 1);
		rd.ComputeListEnd();

		// Submit to GPU and wait for sync
		rd.Submit();
		rd.Sync();

		var computeList1 = rd.ComputeListBegin();
		rd.ComputeListBindComputePipeline(computeList1, pipeline);
		rd.ComputeListBindUniformSet(computeList1, uniformSet1, 0);
		rd.ComputeListDispatch(computeList1, xGroups: hiddenSize, yGroups: inBatches, zGroups: 1);
		rd.ComputeListEnd();

		// Submit to GPU and wait for sync
		rd.Submit();
		rd.Sync();

		var computeList2 = rd.ComputeListBegin();
		rd.ComputeListBindComputePipeline(computeList2, pipeline);
		rd.ComputeListBindUniformSet(computeList2, uniformSet2, 0);
		rd.ComputeListDispatch(computeList2, xGroups: outFeatureSize, yGroups: inBatches, zGroups: 1);
		rd.ComputeListEnd();

		// Submit to GPU and wait for sync
		rd.Submit();
		rd.Sync();
		
		// Read back the data from the buffers
		outputBytes = rd.BufferGetData(buffer9);
		output = new float[inBatches * outFeatureSize];
		Buffer.BlockCopy(outputBytes, 0, output, 0, outputBytes.Length);

		return output;
	}
	
	public override void _ExitTree()
	{
		for(int i = 0; i < storageBuffers.Length; i++) {
			rd.FreeRid(storageBuffers[i]);
		}
		rd.FreeRid(shader);
		rd.Free();
	}
}
