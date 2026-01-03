using Godot;
using System;

public partial class RenderSlangHelloWorld : Node3D
{
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		var rd = RenderingServer.CreateLocalRenderingDevice();
		
		// Load GLSL shader
		var shaderFile = GD.Load<RDShaderFile>("res://SlangHelloWorld/hello-world.glsl");
		var shaderBytecode = shaderFile.GetSpirV();
		var shader = rd.ShaderCreateFromSpirV(shaderBytecode);
		
		// Prepare our data. We use floats in the shader, so we need 32 bit.
		float[] input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
		var inputBytes = new byte[input.Length * sizeof(float)];
		Buffer.BlockCopy(input, 0, inputBytes, 0, inputBytes.Length);

		// Create a storage buffer that can hold our float values.
		// Each float has 4 bytes (32 bit) so 10 x 4 = 40 bytes
		var buffer0 = rd.StorageBufferCreate((uint)inputBytes.Length, inputBytes);
		var buffer1 = rd.StorageBufferCreate((uint)inputBytes.Length, inputBytes);
		var buffer2 = rd.StorageBufferCreate((uint)inputBytes.Length, inputBytes);

		// Create a uniform to assign the buffer to the rendering device
		var buffer0_0 = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 0
		};
		var buffer1_0 = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 1
		};
		var result_0 = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 2
		};
		buffer0_0.AddId(buffer0);
		buffer1_0.AddId(buffer1);
		result_0.AddId(buffer2);

		var uniformSet = rd.UniformSetCreate([buffer0_0, buffer1_0, result_0], shader, 0);
		
		// Create a compute pipeline
		var pipeline = rd.ComputePipelineCreate(shader);
		var computeList = rd.ComputeListBegin();
		rd.ComputeListBindComputePipeline(computeList, pipeline);
		rd.ComputeListBindUniformSet(computeList, uniformSet, 0);
		rd.ComputeListDispatch(computeList, xGroups: 10, yGroups: 1, zGroups: 1);
		rd.ComputeListEnd();
		
		// Submit to GPU and wait for sync
		rd.Submit();
		rd.Sync();
		
		// Read back the data from the buffers
		var outputBytes = rd.BufferGetData(buffer2);
		var output = new float[input.Length];
		Buffer.BlockCopy(outputBytes, 0, output, 0, outputBytes.Length);
		GD.Print("Input: ", string.Join(", ", input));
		GD.Print("Output: ", string.Join(", ", output));
		
		rd.Free();
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}
}
