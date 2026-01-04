using Godot;
using System;
using System.IO;

public struct NeuralNetLinearLayer {
	public float[] weights;
	public float[] bias;
}
public struct NeuralNetLunarLander {
	public NeuralNetLinearLayer[] layers;
	public NeuralNetLunarLander() {
		layers = new NeuralNetLinearLayer[3];
	} 
}

public class NeuralNetModelFile
{
	static float[] ReadTensor(BinaryReader br)
	{
		int count = br.ReadInt32();     // element count
		float[] data = new float[count];

		byte[] bytes = br.ReadBytes(count * sizeof(float));
		Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);

		return data;
	}
	
	public static NeuralNetLunarLander ReadFile(string filepath) {
		NeuralNetLunarLander nn = new NeuralNetLunarLander();
		using (BinaryReader br = new BinaryReader(File.Open(filepath, FileMode.Open)))
		{
			nn.layers[0].weights =  ReadTensor(br);
			nn.layers[0].bias =  ReadTensor(br);

			nn.layers[1].weights =  ReadTensor(br);
			nn.layers[1].bias =  ReadTensor(br);

			nn.layers[2].weights =  ReadTensor(br);
			nn.layers[2].bias =  ReadTensor(br);
		}
		return nn;
	}

}
