static inline void tlb_set_dirty(CPUState *env, target_ulong vaddr)

{

    int i;



    vaddr &= TARGET_PAGE_MASK;

    i = (vaddr >> TARGET_PAGE_BITS) & (CPU_TLB_SIZE - 1);

    tlb_set_dirty1(&env->tlb_table[0][i], vaddr);

    tlb_set_dirty1(&env->tlb_table[1][i], vaddr);

#if (NB_MMU_MODES >= 3)

    tlb_set_dirty1(&env->tlb_table[2][i], vaddr);

#endif

#if (NB_MMU_MODES >= 4)

    tlb_set_dirty1(&env->tlb_table[3][i], vaddr);

#endif

#if (NB_MMU_MODES >= 5)

    tlb_set_dirty1(&env->tlb_table[4][i], vaddr);

#endif

}
