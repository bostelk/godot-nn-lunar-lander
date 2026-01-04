using Godot;
using System;

public partial class LanderController : Node3D
{
	[Export]
	public RenderNeuralNetLunarLander ControlNet;

	[Export]
	public RigidBody3D Body;

	float x;
	float y;
	float velX;
	float velY;
	float angle;
	float angularVel;
	float leftLegContact;
	float rightLegContact;

	enum Action {
		DoNothing,
		FireLeftEngine,
		FireMainEngine,
		FireRightEngine
	}

	float[] Low = {
		-10, -10, -10, -10, -10, -10, 0, 0
	};
	float[] High = {
		 10,  10,  10,  10,  10,  10, 1, 1
	};

	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		Body.ContactMonitor = true;
		Body.MaxContactsReported = 1;
		Body.BodyEntered += OnBodyEntered;
		Body.BodyExited += OnBodyExited;
	}
	
	private void OnBodyEntered(Node body)
	{
		leftLegContact = 1;
		rightLegContact = 1;
	}

	private void OnBodyExited(Node body)
	{
		leftLegContact = 0;
		rightLegContact = 0;
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
		Observe();

		var observations = GetObservations();
		Box(observations, Low, High);
		GD.Print(" ", string.Join(", ", observations));
		
		var predictions = ControlNet.PredictSync(observations);

		const float SideEnginePower = 0.2f;
		const float MainEnginePower = 0.2f;

		Action action = (Action)TopPredictionIndex(predictions);
		switch(action) {
			case Action.FireLeftEngine: {
				Body.ApplyImpulse(new Vector3(SideEnginePower,0,0));
				break;
			}
			case Action.FireMainEngine: {
				Body.ApplyImpulse(new Vector3(0,MainEnginePower,0));
				break;
			}
			case Action.FireRightEngine: {
				Body.ApplyImpulse(new Vector3(-SideEnginePower,0,0));
				break;
			}
		}
		
		GD.Print(action);
	}

	void Observe()
	{
		x = Body.Position.X;
		y = Body.Position.Y;
		velX = Body.LinearVelocity.X;
		velY = Body.LinearVelocity.Y;
		angle = Body.Rotation.Z;
		angularVel = Body.AngularVelocity.Z;
	}

	float[] GetObservations()
	{
		float[] state = new float[] { x, y, velX, velY, angle, angularVel, leftLegContact, rightLegContact };
		return state;
	}

	void Box(float[] values, float[] low, float[] high)
	{
		for(int i = 0; i < values.Length; i++)
		{
			if (values[i] < low[i]) {
				values[i] = low[i];
			}
			else if (values[i] > high[i]) {
				values[i] = low[i];
			}
		}
	}

	int TopPredictionIndex(float[] arr)
	{
		int index = -1;
		float value = float.MinValue;

		for(int i = 0; i < arr.Length; i++) 
		{
			if (arr[i] > value) {
				value = arr[i];
				index = i;
			}
		}

		return index;
	}
}
