uint64_t helper_fadd(CPUPPCState *env, uint64_t arg1, uint64_t arg2)

{

    CPU_DoubleU farg1, farg2;



    farg1.ll = arg1;

    farg2.ll = arg2;



    if (unlikely(float64_is_infinity(farg1.d) && float64_is_infinity(farg2.d) &&

                 float64_is_neg(farg1.d) != float64_is_neg(farg2.d))) {

        /* Magnitude subtraction of infinities */

        farg1.ll = fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXISI);

    } else {

        if (unlikely(float64_is_signaling_nan(farg1.d) ||

                     float64_is_signaling_nan(farg2.d))) {

            /* sNaN addition */

            fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN);

        }

        farg1.d = float64_add(farg1.d, farg2.d, &env->fp_status);

    }



    return farg1.ll;

}
