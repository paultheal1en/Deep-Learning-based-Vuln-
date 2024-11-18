float32 HELPER(ucf64_divs)(float32 a, float32 b, CPUUniCore32State *env)

{

    return float32_div(a, b, &env->ucf64.fp_status);

}
