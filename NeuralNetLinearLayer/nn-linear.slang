// nn-linear.slang
StructuredBuffer<float> input;
StructuredBuffer<float> weights;
StructuredBuffer<float> bias;
RWStructuredBuffer<float> output;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{   
    const uint inFeatureSize = 5;
    const uint outFeatureSize = 10;

    uint index = threadId.x;
    uint batch = threadId.y;

    // affine transform
    float sum = 0;
    for(int j = 0; j < inFeatureSize; j++) {
        sum += input[batch * inFeatureSize + j] * weights[index * inFeatureSize + j];
    }
    sum += bias[index];

    output[batch * outFeatureSize + index] = sum;
}
