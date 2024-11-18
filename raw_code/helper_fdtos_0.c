float32 helper_fdtos(CPUSPARCState *env, float64 src)

{

    float32 ret;

    clear_float_exceptions(env);

    ret = float64_to_float32(src, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
