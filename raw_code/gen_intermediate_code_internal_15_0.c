static inline void gen_intermediate_code_internal(TranslationBlock * tb,

                                                  int spc, CPUSPARCState *env)

{

    target_ulong pc_start, last_pc;

    uint16_t *gen_opc_end;

    DisasContext dc1, *dc = &dc1;

    CPUBreakpoint *bp;

    int j, lj = -1;

    int num_insns;

    int max_insns;



    memset(dc, 0, sizeof(DisasContext));

    dc->tb = tb;

    pc_start = tb->pc;

    dc->pc = pc_start;

    last_pc = dc->pc;

    dc->npc = (target_ulong) tb->cs_base;

    dc->cc_op = CC_OP_DYNAMIC;

    dc->mem_idx = cpu_mmu_index(env);

    dc->def = env->def;

    if ((dc->def->features & CPU_FEATURE_FLOAT))

        dc->fpu_enabled = cpu_fpu_enabled(env);

    else

        dc->fpu_enabled = 0;

#ifdef TARGET_SPARC64

    dc->address_mask_32bit = env->pstate & PS_AM;

#endif

    gen_opc_end = gen_opc_buf + OPC_MAX_SIZE;



    cpu_tmp0 = tcg_temp_new();

    cpu_tmp32 = tcg_temp_new_i32();

    cpu_tmp64 = tcg_temp_new_i64();



    cpu_dst = tcg_temp_local_new();



    // loads and stores

    cpu_val = tcg_temp_local_new();

    cpu_addr = tcg_temp_local_new();



    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0)

        max_insns = CF_COUNT_MASK;

    gen_icount_start();

    do {

        if (unlikely(!TAILQ_EMPTY(&env->breakpoints))) {

            TAILQ_FOREACH(bp, &env->breakpoints, entry) {

                if (bp->pc == dc->pc) {

                    if (dc->pc != pc_start)

                        save_state(dc, cpu_cond);

                    gen_helper_debug();

                    tcg_gen_exit_tb(0);

                    dc->is_br = 1;

                    goto exit_gen_loop;

                }

            }

        }

        if (spc) {

            qemu_log("Search PC...\n");

            j = gen_opc_ptr - gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j)

                    gen_opc_instr_start[lj++] = 0;

                gen_opc_pc[lj] = dc->pc;

                gen_opc_npc[lj] = dc->npc;

                gen_opc_instr_start[lj] = 1;

                gen_opc_icount[lj] = num_insns;

            }

        }

        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO))

            gen_io_start();

        last_pc = dc->pc;

        disas_sparc_insn(dc);

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

        if (env->singlestep_enabled || singlestep) {

            tcg_gen_movi_tl(cpu_pc, dc->pc);

            tcg_gen_exit_tb(0);

            break;

        }

    } while ((gen_opc_ptr < gen_opc_end) &&

             (dc->pc - pc_start) < (TARGET_PAGE_SIZE - 32) &&

             num_insns < max_insns);



 exit_gen_loop:

    tcg_temp_free(cpu_addr);

    tcg_temp_free(cpu_val);

    tcg_temp_free(cpu_dst);

    tcg_temp_free_i64(cpu_tmp64);

    tcg_temp_free_i32(cpu_tmp32);

    tcg_temp_free(cpu_tmp0);

    if (tb->cflags & CF_LAST_IO)

        gen_io_end();

    if (!dc->is_br) {

        if (dc->pc != DYNAMIC_PC &&

            (dc->npc != DYNAMIC_PC && dc->npc != JUMP_PC)) {

            /* static PC and NPC: we can use direct chaining */

            gen_goto_tb(dc, 0, dc->pc, dc->npc);

        } else {

            if (dc->pc != DYNAMIC_PC)

                tcg_gen_movi_tl(cpu_pc, dc->pc);

            save_npc(dc, cpu_cond);

            tcg_gen_exit_tb(0);

        }

    }

    gen_icount_end(tb, num_insns);

    *gen_opc_ptr = INDEX_op_end;

    if (spc) {

        j = gen_opc_ptr - gen_opc_buf;

        lj++;

        while (lj <= j)

            gen_opc_instr_start[lj++] = 0;

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

        log_target_disas(pc_start, last_pc + 4 - pc_start, 0);

        qemu_log("\n");

    }

#endif

}
