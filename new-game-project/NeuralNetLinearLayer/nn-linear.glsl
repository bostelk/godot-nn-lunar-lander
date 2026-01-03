#[compute]
#version 450

layout(row_major) uniform;
layout(row_major) buffer;

#line 2 0
layout(std430, binding = 0) readonly buffer StructuredBuffer_float_t_0 {
    float _data[];
} input_0;

#line 3
layout(std430, binding = 1) readonly buffer StructuredBuffer_float_t_1 {
    float _data[];
} weights_0;

#line 4
layout(std430, binding = 2) readonly buffer StructuredBuffer_float_t_2 {
    float _data[];
} bias_0;

#line 5
layout(std430, binding = 3) buffer StructuredBuffer_float_t_3 {
    float _data[];
} output_0;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{


    uint index_0 = gl_GlobalInvocationID.x;
    uint batch_0 = gl_GlobalInvocationID.y;

#line 15
    int j_0 = 0;

#line 15
    float sum_0 = 0.0;



    for(;;)
    {

#line 19
        uint _S1 = uint(j_0);

#line 19
        if(_S1 < 5U)
        {
        }
        else
        {

#line 19
            break;
        }

#line 20
        float sum_1 = sum_0 + input_0._data[uint(batch_0 * 5U + _S1)] * weights_0._data[uint(index_0 * 5U + _S1)];

#line 19
        j_0 = j_0 + 1;

#line 19
        sum_0 = sum_1;

#line 19
    }

#line 24
    output_0._data[uint(batch_0 * 10U + index_0)] = sum_0 + bias_0._data[uint(index_0)];
    return;
}

