float64 helper_fstod(CPUSPARCState *env, float32 src)

{

    float64 ret;

    clear_float_exceptions(env);

    ret = float32_to_float64(src, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
