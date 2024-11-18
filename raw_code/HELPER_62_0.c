float32 HELPER(ucf64_si2sf)(float32 x, CPUUniCore32State *env)

{

    return int32_to_float32(ucf64_stoi(x), &env->ucf64.fp_status);

}
