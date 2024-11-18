void helper_xssubqp(CPUPPCState *env, uint32_t opcode)

{

    ppc_vsr_t xt, xa, xb;

    float_status tstat;



    getVSR(rA(opcode) + 32, &xa, env);

    getVSR(rB(opcode) + 32, &xb, env);

    getVSR(rD(opcode) + 32, &xt, env);

    helper_reset_fpstatus(env);



    if (unlikely(Rc(opcode) != 0)) {

        /* TODO: Support xssubqp after round-to-odd is implemented */

        abort();

    }



    tstat = env->fp_status;

    set_float_exception_flags(0, &tstat);

    xt.f128 = float128_sub(xa.f128, xb.f128, &tstat);

    env->fp_status.float_exception_flags |= tstat.float_exception_flags;



    if (unlikely(tstat.float_exception_flags & float_flag_invalid)) {

        if (float128_is_infinity(xa.f128) && float128_is_infinity(xb.f128)) {

            float_invalid_op_excp(env, POWERPC_EXCP_FP_VXISI, 1);

        } else if (float128_is_signaling_nan(xa.f128, &tstat) ||

                   float128_is_signaling_nan(xb.f128, &tstat)) {

            float_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN, 1);

        }

    }



    helper_compute_fprf_float128(env, xt.f128);

    putVSR(rD(opcode) + 32, &xt, env);

    float_check_status(env);

}
