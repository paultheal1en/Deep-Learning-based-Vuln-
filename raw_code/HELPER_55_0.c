float64 HELPER(ucf64_addd)(float64 a, float64 b, CPUUniCore32State *env)

{

    return float64_add(a, b, &env->ucf64.fp_status);

}
