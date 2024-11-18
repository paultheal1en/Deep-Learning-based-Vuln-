uint64_t helper_fres(CPUPPCState *env, uint64_t arg)

{

    CPU_DoubleU farg;

    float32 f32;



    farg.ll = arg;



    if (unlikely(float64_is_signaling_nan(farg.d))) {

        /* sNaN reciprocal */

        fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN);

    }

    farg.d = float64_div(float64_one, farg.d, &env->fp_status);

    f32 = float64_to_float32(farg.d, &env->fp_status);

    farg.d = float32_to_float64(f32, &env->fp_status);



    return farg.ll;

}
