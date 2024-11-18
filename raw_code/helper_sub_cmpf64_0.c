float64 helper_sub_cmpf64(CPUM68KState *env, float64 src0, float64 src1)

{

    /* ??? This may incorrectly raise exceptions.  */

    /* ??? Should flush denormals to zero.  */

    float64 res;

    res = float64_sub(src0, src1, &env->fp_status);

    if (float64_is_nan(res)) {

        /* +/-inf compares equal against itself, but sub returns nan.  */

        if (!float64_is_nan(src0)

            && !float64_is_nan(src1)) {

            res = 0;

            if (float64_lt_quiet(src0, res, &env->fp_status))

                res = float64_chs(res);

        }

    }

    return res;

}
