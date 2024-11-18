target_ulong helper_rdhwr_xnp(CPUMIPSState *env)

{

    check_hwrena(env, 5);

    return (env->CP0_Config5 >> CP0C5_XNP) & 1;

}
