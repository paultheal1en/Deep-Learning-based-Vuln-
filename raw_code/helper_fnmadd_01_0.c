uint64_t helper_fnmadd(CPUPPCState *env, uint64_t arg1, uint64_t arg2,

                       uint64_t arg3)

{

    CPU_DoubleU farg1, farg2, farg3;



    farg1.ll = arg1;

    farg2.ll = arg2;

    farg3.ll = arg3;



    if (unlikely((float64_is_infinity(farg1.d) && float64_is_zero(farg2.d)) ||

                 (float64_is_zero(farg1.d) && float64_is_infinity(farg2.d)))) {

        /* Multiplication of zero by infinity */

        farg1.ll = fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXIMZ);

    } else {

        if (unlikely(float64_is_signaling_nan(farg1.d) ||

                     float64_is_signaling_nan(farg2.d) ||

                     float64_is_signaling_nan(farg3.d))) {

            /* sNaN operation */

            fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN);

        }

        /* This is the way the PowerPC specification defines it */

        float128 ft0_128, ft1_128;



        ft0_128 = float64_to_float128(farg1.d, &env->fp_status);

        ft1_128 = float64_to_float128(farg2.d, &env->fp_status);

        ft0_128 = float128_mul(ft0_128, ft1_128, &env->fp_status);

        if (unlikely(float128_is_infinity(ft0_128) &&

                     float64_is_infinity(farg3.d) &&

                     float128_is_neg(ft0_128) != float64_is_neg(farg3.d))) {

            /* Magnitude subtraction of infinities */

            farg1.ll = fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXISI);

        } else {

            ft1_128 = float64_to_float128(farg3.d, &env->fp_status);

            ft0_128 = float128_add(ft0_128, ft1_128, &env->fp_status);

            farg1.d = float128_to_float64(ft0_128, &env->fp_status);

        }

        if (likely(!float64_is_any_nan(farg1.d))) {

            farg1.d = float64_chs(farg1.d);

        }

    }

    return farg1.ll;

}
