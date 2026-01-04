#[compute]
#version 450
layout(row_major) uniform;
layout(row_major) buffer;

#line 5 0
layout(std430, binding = 1) readonly buffer StructuredBuffer_float_t_0 {
    float _data[];
} input_0;
layout(std430, binding = 4) buffer StructuredBuffer_float_t_1 {
    float _data[];
} output_0;

#line 6
layout(std430, binding = 2) readonly buffer StructuredBuffer_float_t_2 {
    float _data[];
} weights_0;

#line 7
layout(std430, binding = 3) readonly buffer StructuredBuffer_float_t_3 {
    float _data[];
} bias_0;

#line 90 1
struct GlobalParams_0
{
    int inSize_0;
    int outSize_0;
    int relu_0;
};


#line 90
layout(binding = 0)
layout(std140) uniform block_GlobalParams_0
{
    int inSize_0;
    int outSize_0;
    int relu_0;
}globalParams_0;

#line 27 0
void linear_0(uint _S1, uint _S2, uint _S3, uint _S4, bool _S5)
{

#line 27
    int j_0 = 0;

#line 27
    float sum_0 = 0.0;

#line 13
    for(;;)
    {

#line 13
        uint _S6 = uint(j_0);

#line 13
        if(_S6 < _S1)
        {
        }
        else
        {

#line 13
            break;
        }

#line 14
        float sum_1 = sum_0 + input_0._data[uint(_S3 * _S1 + _S6)] * weights_0._data[uint(_S4 * _S1 + _S6)];

#line 13
        j_0 = j_0 + 1;

#line 13
        sum_0 = sum_1;

#line 13
    }


    float sum_2 = sum_0 + bias_0._data[uint(_S4)];

    if(_S5)
    {

#line 18
        sum_0 = max(0.0, sum_2);

#line 18
    }
    else
    {

#line 18
        sum_0 = sum_2;

#line 18
    }



    output_0._data[uint(_S3 * _S2 + _S4)] = sum_0;
    return;
}


layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{

#line 27
    linear_0(uint(globalParams_0.inSize_0), uint(globalParams_0.outSize_0), gl_GlobalInvocationID.y, gl_GlobalInvocationID.x, (globalParams_0.relu_0) == 255);

#line 33
    return;
}

