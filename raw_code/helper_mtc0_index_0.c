void helper_mtc0_index(CPUMIPSState *env, target_ulong arg1)

{

    int num = 1;

    unsigned int tmp = env->tlb->nb_tlb;



    do {

        tmp >>= 1;

        num <<= 1;

    } while (tmp);

    env->CP0_Index = (env->CP0_Index & 0x80000000) | (arg1 & (num - 1));

}
