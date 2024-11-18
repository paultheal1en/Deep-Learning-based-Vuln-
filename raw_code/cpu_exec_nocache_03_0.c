static void cpu_exec_nocache(CPUArchState *env, int max_cycles,

                             TranslationBlock *orig_tb)

{

    CPUState *cpu = ENV_GET_CPU(env);

    TranslationBlock *tb;

    target_ulong pc = orig_tb->pc;

    target_ulong cs_base = orig_tb->cs_base;

    uint64_t flags = orig_tb->flags;



    /* Should never happen.

       We only end up here when an existing TB is too long.  */

    if (max_cycles > CF_COUNT_MASK)

        max_cycles = CF_COUNT_MASK;



    /* tb_gen_code can flush our orig_tb, invalidate it now */

    tb_phys_invalidate(orig_tb, -1);

    tb = tb_gen_code(cpu, pc, cs_base, flags,

                     max_cycles);

    cpu->current_tb = tb;

    /* execute the generated code */

    trace_exec_tb_nocache(tb, tb->pc);

    cpu_tb_exec(cpu, tb->tc_ptr);

    cpu->current_tb = NULL;

    tb_phys_invalidate(tb, -1);

    tb_free(tb);

}
