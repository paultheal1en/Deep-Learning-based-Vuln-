target_ulong helper_rdhwr_performance(CPUMIPSState *env)

{

    check_hwrena(env, 4);

    return env->CP0_Performance0;

}
