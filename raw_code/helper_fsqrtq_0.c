void helper_fsqrtq(CPUSPARCState *env)

{

    clear_float_exceptions(env);

    QT0 = float128_sqrt(QT1, &env->fp_status);

    check_ieee_exceptions(env);

}
