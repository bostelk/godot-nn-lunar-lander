using Godot;
using System;

public partial class RenderComputeExample : Node3D
{
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		var rd = RenderingServer.CreateLocalRenderingDevice();
		
		// Load GLSL shader
		var shaderFile = GD.Load<RDShaderFile>("res://GodotComputeExample/compute_example.glsl");
		var shaderBytecode = shaderFile.GetSpirV();
		var shader = rd.ShaderCreateFromSpirV(shaderBytecode);
		
		// Prepare our data. We use floats in the shader, so we need 32 bit.
		float[] input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
		var inputBytes = new byte[input.Length * sizeof(float)];
		Buffer.BlockCopy(input, 0, inputBytes, 0, inputBytes.Length);

		// Create a storage buffer that can hold our float values.
		// Each float has 4 bytes (32 bit) so 10 x 4 = 40 bytes
		var buffer = rd.StorageBufferCreate((uint)inputBytes.Length, inputBytes);
		
		// Create a uniform to assign the buffer to the rendering device
		var uniform = new RDUniform
		{
			UniformType = RenderingDevice.UniformType.StorageBuffer,
			Binding = 0
		};
		uniform.AddId(buffer);
		var uniformSet = rd.UniformSetCreate([uniform], shader, 0);
		
		// Create a compute pipeline
		var pipeline = rd.ComputePipelineCreate(shader);
		var computeList = rd.ComputeListBegin();
		rd.ComputeListBindComputePipeline(computeList, pipeline);
		rd.ComputeListBindUniformSet(computeList, uniformSet, 0);
		rd.ComputeListDispatch(computeList, xGroups: 5, yGroups: 1, zGroups: 1);
		rd.ComputeListEnd();
		
		// Submit to GPU and wait for sync
		rd.Submit();
		rd.Sync();
		
		// Read back the data from the buffers
		var outputBytes = rd.BufferGetData(buffer);
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
