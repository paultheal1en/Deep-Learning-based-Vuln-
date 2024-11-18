uint64_t helper_fsel (uint64_t arg1, uint64_t arg2, uint64_t arg3)

{

    CPU_DoubleU farg1;



    farg1.ll = arg1;



    if ((!float64_is_neg(farg1.d) || float64_is_zero(farg1.d)) && !float64_is_nan(farg1.d))

        return arg2;

    else

        return arg3;

}
