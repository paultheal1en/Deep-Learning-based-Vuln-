uint32_t helper_float_cvtw_s(CPUMIPSState *env, uint32_t fst0)

{

    uint32_t wt2;



    wt2 = float32_to_int32(fst0, &env->active_fpu.fp_status);

    update_fcr31(env, GETPC());

    if (get_float_exception_flags(&env->active_fpu.fp_status)

        & (float_flag_invalid | float_flag_overflow)) {

        wt2 = FP_TO_INT32_OVERFLOW;

    }

    return wt2;

}
