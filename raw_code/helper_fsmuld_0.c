float64 helper_fsmuld(CPUSPARCState *env, float32 src1, float32 src2)

{

    float64 ret;

    clear_float_exceptions(env);

    ret = float64_mul(float32_to_float64(src1, &env->fp_status),

                      float32_to_float64(src2, &env->fp_status),

                      &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
