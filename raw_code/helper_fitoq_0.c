void helper_fitoq(CPUSPARCState *env, int32_t src)

{

    /* No possible exceptions converting int to long double.  */

    QT0 = int32_to_float128(src, &env->fp_status);

}
