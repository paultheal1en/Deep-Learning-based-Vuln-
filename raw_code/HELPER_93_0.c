float32 HELPER(ucf64_subs)(float32 a, float32 b, CPUUniCore32State *env)

{

    return float32_sub(a, b, &env->ucf64.fp_status);

}
