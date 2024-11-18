float32 helper_fqtos(CPUSPARCState *env)

{

    float32 ret;

    clear_float_exceptions(env);

    ret = float128_to_float32(QT1, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
