float32 HELPER(ucf64_sf2si)(float32 x, CPUUniCore32State *env)

{

    return ucf64_itos(float32_to_int32(x, &env->ucf64.fp_status));

}
