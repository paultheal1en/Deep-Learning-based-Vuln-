static int cpu_restore_state_from_tb(CPUState *cpu, TranslationBlock *tb,

                                     uintptr_t searched_pc)

{

    CPUArchState *env = cpu->env_ptr;

    TCGContext *s = &tcg_ctx;

    int j;

    uintptr_t tc_ptr;

#ifdef CONFIG_PROFILER

    int64_t ti;

#endif



#ifdef CONFIG_PROFILER

    ti = profile_getclock();

#endif

    tcg_func_start(s);



    gen_intermediate_code_pc(env, tb);



    if (use_icount) {

        /* Reset the cycle counter to the start of the block.  */

        cpu->icount_decr.u16.low += tb->icount;

        /* Clear the IO flag.  */

        cpu->can_do_io = 0;

    }



    /* find opc index corresponding to search_pc */

    tc_ptr = (uintptr_t)tb->tc_ptr;

    if (searched_pc < tc_ptr)

        return -1;



    s->tb_next_offset = tb->tb_next_offset;

#ifdef USE_DIRECT_JUMP

    s->tb_jmp_offset = tb->tb_jmp_offset;

    s->tb_next = NULL;

#else

    s->tb_jmp_offset = NULL;

    s->tb_next = tb->tb_next;

#endif

    j = tcg_gen_code_search_pc(s, (tcg_insn_unit *)tc_ptr,

                               searched_pc - tc_ptr);

    if (j < 0)

        return -1;

    /* now find start of instruction before */

    while (s->gen_opc_instr_start[j] == 0) {

        j--;

    }

    cpu->icount_decr.u16.low -= s->gen_opc_icount[j];



    restore_state_to_opc(env, tb, j);



#ifdef CONFIG_PROFILER

    s->restore_time += profile_getclock() - ti;

    s->restore_count++;

#endif

    return 0;

}
