float32 HELPER(ucf64_df2si)(float64 x, CPUUniCore32State *env)

{

    return ucf64_itos(float64_to_int32(x, &env->ucf64.fp_status));

}
