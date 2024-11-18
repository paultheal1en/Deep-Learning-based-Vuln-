void helper_fstoq(CPUSPARCState *env, float32 src)

{

    clear_float_exceptions(env);

    QT0 = float32_to_float128(src, &env->fp_status);

    check_ieee_exceptions(env);

}
