static inline void gen_intermediate_code_internal(SPARCCPU *cpu,

                                                  TranslationBlock *tb,

                                                  bool spc)

{

    CPUState *cs = CPU(cpu);

    CPUSPARCState *env = &cpu->env;

    target_ulong pc_start, last_pc;

    uint16_t *gen_opc_end;

    DisasContext dc1, *dc = &dc1;

    CPUBreakpoint *bp;

    int j, lj = -1;

    int num_insns;

    int max_insns;

    unsigned int insn;



    memset(dc, 0, sizeof(DisasContext));

    dc->tb = tb;

    pc_start = tb->pc;

    dc->pc = pc_start;

    last_pc = dc->pc;

    dc->npc = (target_ulong) tb->cs_base;

    dc->cc_op = CC_OP_DYNAMIC;

    dc->mem_idx = cpu_mmu_index(env);

    dc->def = env->def;

    dc->fpu_enabled = tb_fpu_enabled(tb->flags);

    dc->address_mask_32bit = tb_am_enabled(tb->flags);

    dc->singlestep = (cs->singlestep_enabled || singlestep);

    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;



    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0)

        max_insns = CF_COUNT_MASK;

    gen_tb_start();

    do {

        if (unlikely(!QTAILQ_EMPTY(&cs->breakpoints))) {

            QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

                if (bp->pc == dc->pc) {

                    if (dc->pc != pc_start)

                        save_state(dc);

                    gen_helper_debug(cpu_env);

                    tcg_gen_exit_tb(0);

                    dc->is_br = 1;

                    goto exit_gen_loop;

                }

            }

        }

        if (spc) {

            qemu_log("Search PC...\n");

            j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j)

                    tcg_ctx.gen_opc_instr_start[lj++] = 0;

                tcg_ctx.gen_opc_pc[lj] = dc->pc;

                gen_opc_npc[lj] = dc->npc;

                tcg_ctx.gen_opc_instr_start[lj] = 1;

                tcg_ctx.gen_opc_icount[lj] = num_insns;

            }

        }

        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO))

            gen_io_start();

        last_pc = dc->pc;

        insn = cpu_ldl_code(env, dc->pc);



        disas_sparc_insn(dc, insn);

        num_insns++;



        if (dc->is_br)

            break;

        /* if the next PC is different, we abort now */

        if (dc->pc != (last_pc + 4))

            break;

        /* if we reach a page boundary, we stop generation so that the

           PC of a TT_TFAULT exception is always in the right page */

        if ((dc->pc & (TARGET_PAGE_SIZE - 1)) == 0)

            break;

        /* if single step mode, we generate only one instruction and

           generate an exception */

        if (dc->singlestep) {

            break;

        }

    } while ((tcg_ctx.gen_opc_ptr < gen_opc_end) &&

             (dc->pc - pc_start) < (TARGET_PAGE_SIZE - 32) &&

             num_insns < max_insns);



 exit_gen_loop:

    if (tb->cflags & CF_LAST_IO) {

        gen_io_end();

    }

    if (!dc->is_br) {

        if (dc->pc != DYNAMIC_PC &&

            (dc->npc != DYNAMIC_PC && dc->npc != JUMP_PC)) {

            /* static PC and NPC: we can use direct chaining */

            gen_goto_tb(dc, 0, dc->pc, dc->npc);

        } else {

            if (dc->pc != DYNAMIC_PC) {

                tcg_gen_movi_tl(cpu_pc, dc->pc);

            }

            save_npc(dc);

            tcg_gen_exit_tb(0);

        }

    }

    gen_tb_end(tb, num_insns);

    *tcg_ctx.gen_opc_ptr = INDEX_op_end;

    if (spc) {

        j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

        lj++;

        while (lj <= j)

            tcg_ctx.gen_opc_instr_start[lj++] = 0;

#if 0

        log_page_dump();

#endif

        gen_opc_jump_pc[0] = dc->jump_pc[0];

        gen_opc_jump_pc[1] = dc->jump_pc[1];

    } else {

        tb->size = last_pc + 4 - pc_start;

        tb->icount = num_insns;

    }

#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("--------------\n");

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(env, pc_start, last_pc + 4 - pc_start, 0);

        qemu_log("\n");

    }

#endif

}
