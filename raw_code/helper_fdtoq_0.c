void helper_fdtoq(CPUSPARCState *env, float64 src)

{

    clear_float_exceptions(env);

    QT0 = float64_to_float128(src, &env->fp_status);

    check_ieee_exceptions(env);

}
