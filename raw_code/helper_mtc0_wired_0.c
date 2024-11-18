void helper_mtc0_wired(CPUMIPSState *env, target_ulong arg1)

{

    env->CP0_Wired = arg1 % env->tlb->nb_tlb;

}
