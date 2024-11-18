float32 helper_fsqrts(CPUSPARCState *env, float32 src)

{

    float32 ret;

    clear_float_exceptions(env);

    ret = float32_sqrt(src, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
