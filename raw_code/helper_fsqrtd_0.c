float64 helper_fsqrtd(CPUSPARCState *env, float64 src)

{

    float64 ret;

    clear_float_exceptions(env);

    ret = float64_sqrt(src, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
