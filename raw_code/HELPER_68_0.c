float64 HELPER(ucf64_divd)(float64 a, float64 b, CPUUniCore32State *env)

{

    return float64_div(a, b, &env->ucf64.fp_status);

}
