int64_t helper_fqtox(CPUSPARCState *env)

{

    int64_t ret;

    clear_float_exceptions(env);

    ret = float128_to_int64_round_to_zero(QT1, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
