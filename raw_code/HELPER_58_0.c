float32 HELPER(ucf64_df2sf)(float64 x, CPUUniCore32State *env)

{

    return float64_to_float32(x, &env->ucf64.fp_status);

}
