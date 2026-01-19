using Godot;
using System;

public partial class LanderController : Node2D
{
	[Export] 
	public Camera2D Camera;

	[Export]
	public RenderNeuralNetLunarLander ControlNet;

	[Export]
	public RigidBody2D Body;

	[Export]
	public CpuParticles2D MainEngineParticles;
	
	[Export]
	public CpuParticles2D LeftEngineParticles;

	[Export]
	public CpuParticles2D RightEngineParticles;

	float x;
	float y;
	float velX;
	float velY;
	float angle;
	float angularVel;
	float leftLegContact;
	float rightLegContact;

	double contactTime = 0;

	Vector2 left;
	Vector2 down;

	enum Action {
		DoNothing,
		FireLeftEngine,
		FireMainEngine,
		FireRightEngine
	}
	Action userAction;

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

		Reset();
	}

	private void Reset()
	{
		Body.Rotation = 0;

		leftLegContact = 0;
		rightLegContact = 0;
		contactTime = 0;

		// The lander starts at the top-center of the viewport.
		float RandomX = GD.RandRange(-200,200);
		Vector2 position = new Vector2(RandomX, -200);
		Body.Position = position;

		const float InitialPower = 100;
		float Theta = (float)GD.RandRange(0f, 3.14f*2f);
		//Body.ApplyImpulse(new Vector2(InitialPower * (float)Math.Sin(Theta), InitialPower * (float)Math.Cos(Theta)));
	}
	
	private void ResetScene()
	{
		var currentScene = GetTree().CurrentScene;
		GetTree().ReloadCurrentScene();
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

	public override void _Input(InputEvent @event)
	{
		if (@event.IsActionPressed("Reset"))
		{
			ResetScene();
		}
	}

	private Action GetUserAction()
	{
		var userAction = Action.DoNothing;

		if (Input.IsActionPressed("FireLeftEngine"))
		{
			userAction = Action.FireLeftEngine;
		}
		if (Input.IsActionPressed("FireRightEngine"))
		{
			userAction = Action.FireRightEngine;
		}
		if (Input.IsActionPressed("FireMainEngine"))
		{
			userAction = Action.FireMainEngine;
		}

		return userAction;
	}


	public override void _Draw()
	{
		//DrawLine(Body.GlobalPosition, Body.GlobalPosition - 8 * left, Colors.Green, 1.0f);
		//DrawLine(Body.GlobalPosition, Body.GlobalPosition + 10 * down, Colors.Red, 1.0f);
		//DrawLine(Body.GlobalPosition, Body.GlobalPosition +	10 * -down - 8 * left, Colors.Yellow, 1.0f);
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _PhysicsProcess(double delta) 
	{
		//QueueRedraw();

		Observe();

		if (leftLegContact == 1 && rightLegContact == 1) {
			contactTime += delta;
			if (contactTime > 1) {
				ResetScene();
				return;
			}
		}

		var state = GetState();
		Box(state, Low, High);
		GD.Print(" ", string.Join(", ", state));
		
		var predictions = ControlNet.PredictSync(state);

		const float SideEnginePower = 2.5f;
		const float MainEnginePower = 25f;

		LeftEngineParticles.Emitting = false;
		MainEngineParticles.Emitting = false;
		RightEngineParticles.Emitting = false;

		Action action = (Action)TopPredictionIndex(predictions);

		userAction = GetUserAction();

		if (userAction != Action.DoNothing)
		{
			action = userAction;
		}

		switch(action) {
			case Action.FireLeftEngine: {
				Body.ApplyImpulse(left * SideEnginePower, 10 * -down - 8 * left);
				LeftEngineParticles.Emitting = true;
				break;
			}
			case Action.FireMainEngine: {
				Body.ApplyImpulse(-down * MainEnginePower, 10 * down);
				MainEngineParticles.Emitting = true;
				break;
			}
			case Action.FireRightEngine: {
				Body.ApplyImpulse(-left * SideEnginePower, 10 * -down +8 * left);
				RightEngineParticles.Emitting = true;
				break;
			}
		}
		
		GD.Print(action);
	}

	void Observe()
	{
		const float Scale = 30;
		const float ViewportWidth = 600;
		const float ViewportHeight = 400;
		const float HelipadY = 100;
		const float DistToLeg = 10;
		float Fps = Engine.PhysicsTicksPerSecond;

		// Remap from screen [-300,300] to normalized [-1,1]
		x = Body.Position.X / (ViewportWidth / 2);

		// Remap from top, bottom screen [-200,200] to [400, 0]
		y = Body.Position.Y * -1 + ViewportHeight / 2;
		// Remap from top, bottom screen [400, 0] to normalized [1, 0]
		y = (y - DistToLeg - HelipadY) / (ViewportHeight / 2);

		velX = Body.LinearVelocity.X / Fps;
		velY = -Body.LinearVelocity.Y / Fps;
		angle = Body.Rotation;
		angularVel = 20f * Body.AngularVelocity / Fps;

		down = new Vector2(Mathf.Cos(angle + Mathf.Pi/2), Mathf.Sin(angle + Mathf.Pi/2));
		left = new Vector2(-down.Y, down.X);

		angle *=-1;
		angularVel *=-1;
	}

	float[] GetState()
	{
		return new float[] { x, y, velX, velY, angle, angularVel, leftLegContact, rightLegContact };
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
