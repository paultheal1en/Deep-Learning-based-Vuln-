target_ulong helper_rdhwr_synci_step(CPUMIPSState *env)

{

    check_hwrena(env, 1);

    return env->SYNCI_Step;

}
