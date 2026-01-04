// nn-lunar-lander.slang
uniform int inSize;
uniform int outSize;
uniform int relu;
StructuredBuffer<float> input;
StructuredBuffer<float> weights;
StructuredBuffer<float> bias;
RWStructuredBuffer<float> output;

void linear(StructuredBuffer<float> input, RWStructuredBuffer<float> output, StructuredBuffer<float> weights, StructuredBuffer<float> bias, uint inSize, uint outSize, uint batch, uint index, bool relu) 
{
    float sum = 0;
    for(int j = 0; j < inSize; j++) {
        sum += input[batch * inSize + j] * weights[index * inSize + j];
    }
    sum += bias[index];

    if (relu) {
        sum = max(0, sum);
    }

    output[batch * outSize + index] = sum;
}

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    uint batch = threadId.y;

    linear(input, output, weights, bias, inSize, outSize, batch, index, relu == 255);
}
