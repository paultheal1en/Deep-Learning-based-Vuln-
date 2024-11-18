gen_intermediate_code_internal(M68kCPU *cpu, TranslationBlock *tb,

                               bool search_pc)

{

    CPUState *cs = CPU(cpu);

    CPUM68KState *env = &cpu->env;

    DisasContext dc1, *dc = &dc1;

    uint16_t *gen_opc_end;

    CPUBreakpoint *bp;

    int j, lj;

    target_ulong pc_start;

    int pc_offset;

    int num_insns;

    int max_insns;



    /* generate intermediate code */

    pc_start = tb->pc;



    dc->tb = tb;



    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;



    dc->env = env;

    dc->is_jmp = DISAS_NEXT;

    dc->pc = pc_start;

    dc->cc_op = CC_OP_DYNAMIC;

    dc->singlestep_enabled = cs->singlestep_enabled;

    dc->fpcr = env->fpcr;

    dc->user = (env->sr & SR_S) == 0;

    dc->is_mem = 0;

    dc->done_mac = 0;

    lj = -1;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0)

        max_insns = CF_COUNT_MASK;



    gen_tb_start();

    do {

        pc_offset = dc->pc - pc_start;

        gen_throws_exception = NULL;

        if (unlikely(!QTAILQ_EMPTY(&cs->breakpoints))) {

            QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

                if (bp->pc == dc->pc) {

                    gen_exception(dc, dc->pc, EXCP_DEBUG);

                    dc->is_jmp = DISAS_JUMP;

                    break;

                }

            }

            if (dc->is_jmp)

                break;

        }

        if (search_pc) {

            j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j)

                    tcg_ctx.gen_opc_instr_start[lj++] = 0;

            }

            tcg_ctx.gen_opc_pc[lj] = dc->pc;

            tcg_ctx.gen_opc_instr_start[lj] = 1;

            tcg_ctx.gen_opc_icount[lj] = num_insns;

        }

        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO))

            gen_io_start();

        dc->insn_pc = dc->pc;

	disas_m68k_insn(env, dc);

        num_insns++;

    } while (!dc->is_jmp && tcg_ctx.gen_opc_ptr < gen_opc_end &&

             !cs->singlestep_enabled &&

             !singlestep &&

             (pc_offset) < (TARGET_PAGE_SIZE - 32) &&

             num_insns < max_insns);



    if (tb->cflags & CF_LAST_IO)

        gen_io_end();

    if (unlikely(cs->singlestep_enabled)) {

        /* Make sure the pc is updated, and raise a debug exception.  */

        if (!dc->is_jmp) {

            gen_flush_cc_op(dc);

            tcg_gen_movi_i32(QREG_PC, dc->pc);

        }

        gen_helper_raise_exception(cpu_env, tcg_const_i32(EXCP_DEBUG));

    } else {

        switch(dc->is_jmp) {

        case DISAS_NEXT:

            gen_flush_cc_op(dc);

            gen_jmp_tb(dc, 0, dc->pc);

            break;

        default:

        case DISAS_JUMP:

        case DISAS_UPDATE:

            gen_flush_cc_op(dc);

            /* indicate that the hash table must be used to find the next TB */

            tcg_gen_exit_tb(0);

            break;

        case DISAS_TB_JUMP:

            /* nothing more to generate */

            break;

        }

    }

    gen_tb_end(tb, num_insns);

    *tcg_ctx.gen_opc_ptr = INDEX_op_end;



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("----------------\n");

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(env, pc_start, dc->pc - pc_start, 0);

        qemu_log("\n");

    }

#endif

    if (search_pc) {

        j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

        lj++;

        while (lj <= j)

            tcg_ctx.gen_opc_instr_start[lj++] = 0;

    } else {

        tb->size = dc->pc - pc_start;

        tb->icount = num_insns;

    }



    //optimize_flags();

    //expand_target_qops();

}
