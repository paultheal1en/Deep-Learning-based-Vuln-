int32_t helper_fqtoi(CPUSPARCState *env)

{

    int32_t ret;

    clear_float_exceptions(env);

    ret = float128_to_int32_round_to_zero(QT1, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
