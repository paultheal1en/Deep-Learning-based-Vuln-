float64 HELPER(ucf64_sf2df)(float32 x, CPUUniCore32State *env)

{

    return float32_to_float64(x, &env->ucf64.fp_status);

}
