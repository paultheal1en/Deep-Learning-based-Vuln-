void tlb_flush_page(CPUState *env, target_ulong addr)

{

    int i;



#if defined(DEBUG_TLB)

    printf("tlb_flush_page: " TARGET_FMT_lx "\n", addr);

#endif

    /* must reset current TB so that interrupts cannot modify the

       links while we are modifying them */

    env->current_tb = NULL;



    addr &= TARGET_PAGE_MASK;

    i = (addr >> TARGET_PAGE_BITS) & (CPU_TLB_SIZE - 1);

    tlb_flush_entry(&env->tlb_table[0][i], addr);

    tlb_flush_entry(&env->tlb_table[1][i], addr);

#if (NB_MMU_MODES >= 3)

    tlb_flush_entry(&env->tlb_table[2][i], addr);

#if (NB_MMU_MODES == 4)

    tlb_flush_entry(&env->tlb_table[3][i], addr);

#endif

#endif



    tlb_flush_jmp_cache(env, addr);



#ifdef USE_KQEMU

    if (env->kqemu_enabled) {

        kqemu_flush_page(env, addr);

    }

#endif

}
