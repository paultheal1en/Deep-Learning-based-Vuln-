TranslationBlock *tb_gen_code(CPUState *cpu,

                              target_ulong pc, target_ulong cs_base,

                              uint32_t flags, int cflags)

{

    CPUArchState *env = cpu->env_ptr;

    TranslationBlock *tb;

    tb_page_addr_t phys_pc, phys_page2;

    target_ulong virt_page2;

    tcg_insn_unit *gen_code_buf;

    int gen_code_size, search_size;

#ifdef CONFIG_PROFILER

    int64_t ti;

#endif



    phys_pc = get_page_addr_code(env, pc);

    if (use_icount && !(cflags & CF_IGNORE_ICOUNT)) {

        cflags |= CF_USE_ICOUNT;

    }



    tb = tb_alloc(pc);

    if (unlikely(!tb)) {

 buffer_overflow:

        /* flush must be done */

        tb_flush(cpu);

        /* cannot fail at this point */

        tb = tb_alloc(pc);

        assert(tb != NULL);

        /* Don't forget to invalidate previous TB info.  */

        tcg_ctx.tb_ctx.tb_invalidated_flag = 1;

    }



    gen_code_buf = tcg_ctx.code_gen_ptr;

    tb->tc_ptr = gen_code_buf;

    tb->cs_base = cs_base;

    tb->flags = flags;

    tb->cflags = cflags;



#ifdef CONFIG_PROFILER

    tcg_ctx.tb_count1++; /* includes aborted translations because of

                       exceptions */

    ti = profile_getclock();

#endif



    tcg_func_start(&tcg_ctx);



    gen_intermediate_code(env, tb);



    trace_translate_block(tb, tb->pc, tb->tc_ptr);



    /* generate machine code */

    tb->jmp_reset_offset[0] = TB_JMP_RESET_OFFSET_INVALID;

    tb->jmp_reset_offset[1] = TB_JMP_RESET_OFFSET_INVALID;

    tcg_ctx.tb_jmp_reset_offset = tb->jmp_reset_offset;

#ifdef USE_DIRECT_JUMP

    tcg_ctx.tb_jmp_insn_offset = tb->jmp_insn_offset;

    tcg_ctx.tb_jmp_target_addr = NULL;

#else

    tcg_ctx.tb_jmp_insn_offset = NULL;

    tcg_ctx.tb_jmp_target_addr = tb->jmp_target_addr;

#endif



#ifdef CONFIG_PROFILER

    tcg_ctx.tb_count++;

    tcg_ctx.interm_time += profile_getclock() - ti;

    tcg_ctx.code_time -= profile_getclock();

#endif



    /* ??? Overflow could be handled better here.  In particular, we

       don't need to re-do gen_intermediate_code, nor should we re-do

       the tcg optimization currently hidden inside tcg_gen_code.  All

       that should be required is to flush the TBs, allocate a new TB,

       re-initialize it per above, and re-do the actual code generation.  */

    gen_code_size = tcg_gen_code(&tcg_ctx, tb);

    if (unlikely(gen_code_size < 0)) {

        goto buffer_overflow;

    }

    search_size = encode_search(tb, (void *)gen_code_buf + gen_code_size);

    if (unlikely(search_size < 0)) {

        goto buffer_overflow;

    }



#ifdef CONFIG_PROFILER

    tcg_ctx.code_time += profile_getclock();

    tcg_ctx.code_in_len += tb->size;

    tcg_ctx.code_out_len += gen_code_size;

    tcg_ctx.search_out_len += search_size;

#endif



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_OUT_ASM) &&

        qemu_log_in_addr_range(tb->pc)) {

        qemu_log("OUT: [size=%d]\n", gen_code_size);

        log_disas(tb->tc_ptr, gen_code_size);

        qemu_log("\n");

        qemu_log_flush();

    }

#endif



    tcg_ctx.code_gen_ptr = (void *)

        ROUND_UP((uintptr_t)gen_code_buf + gen_code_size + search_size,

                 CODE_GEN_ALIGN);



    /* init jump list */

    assert(((uintptr_t)tb & 3) == 0);

    tb->jmp_list_first = (uintptr_t)tb | 2;

    tb->jmp_list_next[0] = (uintptr_t)NULL;

    tb->jmp_list_next[1] = (uintptr_t)NULL;



    /* init original jump addresses wich has been set during tcg_gen_code() */

    if (tb->jmp_reset_offset[0] != TB_JMP_RESET_OFFSET_INVALID) {

        tb_reset_jump(tb, 0);

    }

    if (tb->jmp_reset_offset[1] != TB_JMP_RESET_OFFSET_INVALID) {

        tb_reset_jump(tb, 1);

    }



    /* check next page if needed */

    virt_page2 = (pc + tb->size - 1) & TARGET_PAGE_MASK;

    phys_page2 = -1;

    if ((pc & TARGET_PAGE_MASK) != virt_page2) {

        phys_page2 = get_page_addr_code(env, virt_page2);

    }

    /* As long as consistency of the TB stuff is provided by tb_lock in user

     * mode and is implicit in single-threaded softmmu emulation, no explicit

     * memory barrier is required before tb_link_page() makes the TB visible

     * through the physical hash table and physical page list.

     */

    tb_link_page(tb, phys_pc, phys_page2);

    return tb;

}
