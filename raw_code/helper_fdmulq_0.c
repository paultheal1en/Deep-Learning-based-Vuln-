void helper_fdmulq(CPUSPARCState *env, float64 src1, float64 src2)

{

    clear_float_exceptions(env);

    QT0 = float128_mul(float64_to_float128(src1, &env->fp_status),

                       float64_to_float128(src2, &env->fp_status),

                       &env->fp_status);

    check_ieee_exceptions(env);

}
