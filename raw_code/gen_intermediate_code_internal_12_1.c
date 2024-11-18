gen_intermediate_code_internal(CPUCRISState *env, TranslationBlock *tb,

                               int search_pc)

{

    uint16_t *gen_opc_end;

    uint32_t pc_start;

    unsigned int insn_len;

    int j, lj;

    struct DisasContext ctx;

    struct DisasContext *dc = &ctx;

    uint32_t next_page_start;

    target_ulong npc;

    int num_insns;

    int max_insns;



    qemu_log_try_set_file(stderr);



    if (env->pregs[PR_VR] == 32) {

        dc->decoder = crisv32_decoder;

        dc->clear_locked_irq = 0;

    } else {

        dc->decoder = crisv10_decoder;

        dc->clear_locked_irq = 1;

    }



    /* Odd PC indicates that branch is rexecuting due to exception in the

     * delayslot, like in real hw.

     */

    pc_start = tb->pc & ~1;

    dc->env = env;

    dc->tb = tb;



    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;



    dc->is_jmp = DISAS_NEXT;

    dc->ppc = pc_start;

    dc->pc = pc_start;

    dc->singlestep_enabled = env->singlestep_enabled;

    dc->flags_uptodate = 1;

    dc->flagx_known = 1;

    dc->flags_x = tb->flags & X_FLAG;

    dc->cc_x_uptodate = 0;

    dc->cc_mask = 0;

    dc->update_cc = 0;

    dc->clear_prefix = 0;



    cris_update_cc_op(dc, CC_OP_FLAGS, 4);

    dc->cc_size_uptodate = -1;



    /* Decode TB flags.  */

    dc->tb_flags = tb->flags & (S_FLAG | P_FLAG | U_FLAG \

            | X_FLAG | PFIX_FLAG);

    dc->delayed_branch = !!(tb->flags & 7);

    if (dc->delayed_branch) {

        dc->jmp = JMP_INDIRECT;

    } else {

        dc->jmp = JMP_NOJMP;

    }



    dc->cpustate_changed = 0;



    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log(

                "srch=%d pc=%x %x flg=%" PRIx64 " bt=%x ds=%u ccs=%x\n"

                "pid=%x usp=%x\n"

                "%x.%x.%x.%x\n"

                "%x.%x.%x.%x\n"

                "%x.%x.%x.%x\n"

                "%x.%x.%x.%x\n",

                search_pc, dc->pc, dc->ppc,

                (uint64_t)tb->flags,

                env->btarget, (unsigned)tb->flags & 7,

                env->pregs[PR_CCS],

                env->pregs[PR_PID], env->pregs[PR_USP],

                env->regs[0], env->regs[1], env->regs[2], env->regs[3],

                env->regs[4], env->regs[5], env->regs[6], env->regs[7],

                env->regs[8], env->regs[9],

                env->regs[10], env->regs[11],

                env->regs[12], env->regs[13],

                env->regs[14], env->regs[15]);

        qemu_log("--------------\n");

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

    }



    next_page_start = (pc_start & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;

    lj = -1;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }



    gen_icount_start();

    do {

        check_breakpoint(env, dc);



        if (search_pc) {

            j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j) {

                    tcg_ctx.gen_opc_instr_start[lj++] = 0;

                }

            }

            if (dc->delayed_branch == 1) {

                tcg_ctx.gen_opc_pc[lj] = dc->ppc | 1;

            } else {

                tcg_ctx.gen_opc_pc[lj] = dc->pc;

            }

            tcg_ctx.gen_opc_instr_start[lj] = 1;

            tcg_ctx.gen_opc_icount[lj] = num_insns;

        }



        /* Pretty disas.  */

        LOG_DIS("%8.8x:\t", dc->pc);



        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO)) {

            gen_io_start();

        }

        dc->clear_x = 1;



        insn_len = dc->decoder(env, dc);

        dc->ppc = dc->pc;

        dc->pc += insn_len;

        if (dc->clear_x) {

            cris_clear_x_flag(dc);

        }



        num_insns++;

        /* Check for delayed branches here. If we do it before

           actually generating any host code, the simulator will just

           loop doing nothing for on this program location.  */

        if (dc->delayed_branch) {

            dc->delayed_branch--;

            if (dc->delayed_branch == 0) {

                if (tb->flags & 7) {

                    t_gen_mov_env_TN(dslot, tcg_const_tl(0));

                }

                if (dc->cpustate_changed || !dc->flagx_known

                    || (dc->flags_x != (tb->flags & X_FLAG))) {

                    cris_store_direct_jmp(dc);

                }



                if (dc->clear_locked_irq) {

                    dc->clear_locked_irq = 0;

                    t_gen_mov_env_TN(locked_irq, tcg_const_tl(0));

                }



                if (dc->jmp == JMP_DIRECT_CC) {

                    int l1;



                    l1 = gen_new_label();

                    cris_evaluate_flags(dc);



                    /* Conditional jmp.  */

                    tcg_gen_brcondi_tl(TCG_COND_EQ,

                               env_btaken, 0, l1);

                    gen_goto_tb(dc, 1, dc->jmp_pc);

                    gen_set_label(l1);

                    gen_goto_tb(dc, 0, dc->pc);

                    dc->is_jmp = DISAS_TB_JUMP;

                    dc->jmp = JMP_NOJMP;

                } else if (dc->jmp == JMP_DIRECT) {

                    cris_evaluate_flags(dc);

                    gen_goto_tb(dc, 0, dc->jmp_pc);

                    dc->is_jmp = DISAS_TB_JUMP;

                    dc->jmp = JMP_NOJMP;

                } else {

                    t_gen_cc_jmp(env_btarget, tcg_const_tl(dc->pc));

                    dc->is_jmp = DISAS_JUMP;

                }

                break;

            }

        }



        /* If we are rexecuting a branch due to exceptions on

           delay slots dont break.  */

        if (!(tb->pc & 1) && env->singlestep_enabled) {

            break;

        }

    } while (!dc->is_jmp && !dc->cpustate_changed

            && tcg_ctx.gen_opc_ptr < gen_opc_end

            && !singlestep

            && (dc->pc < next_page_start)

            && num_insns < max_insns);



    if (dc->clear_locked_irq) {

        t_gen_mov_env_TN(locked_irq, tcg_const_tl(0));

    }



    npc = dc->pc;



        if (tb->cflags & CF_LAST_IO)

            gen_io_end();

    /* Force an update if the per-tb cpu state has changed.  */

    if (dc->is_jmp == DISAS_NEXT

        && (dc->cpustate_changed || !dc->flagx_known

        || (dc->flags_x != (tb->flags & X_FLAG)))) {

        dc->is_jmp = DISAS_UPDATE;

        tcg_gen_movi_tl(env_pc, npc);

    }

    /* Broken branch+delayslot sequence.  */

    if (dc->delayed_branch == 1) {

        /* Set env->dslot to the size of the branch insn.  */

        t_gen_mov_env_TN(dslot, tcg_const_tl(dc->pc - dc->ppc));

        cris_store_direct_jmp(dc);

    }



    cris_evaluate_flags(dc);



    if (unlikely(env->singlestep_enabled)) {

        if (dc->is_jmp == DISAS_NEXT) {

            tcg_gen_movi_tl(env_pc, npc);

        }

        t_gen_raise_exception(EXCP_DEBUG);

    } else {

        switch (dc->is_jmp) {

        case DISAS_NEXT:

            gen_goto_tb(dc, 1, npc);

            break;

        default:

        case DISAS_JUMP:

        case DISAS_UPDATE:

            /* indicate that the hash table must be used

                   to find the next TB */

            tcg_gen_exit_tb(0);

            break;

        case DISAS_SWI:

        case DISAS_TB_JUMP:

            /* nothing more to generate */

            break;

        }

    }

    gen_icount_end(tb, num_insns);

    *tcg_ctx.gen_opc_ptr = INDEX_op_end;

    if (search_pc) {

        j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

        lj++;

        while (lj <= j) {

            tcg_ctx.gen_opc_instr_start[lj++] = 0;

        }

    } else {

        tb->size = dc->pc - pc_start;

        tb->icount = num_insns;

    }



#ifdef DEBUG_DISAS

#if !DISAS_CRIS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        log_target_disas(env, pc_start, dc->pc - pc_start,

                                 dc->env->pregs[PR_VR]);

        qemu_log("\nisize=%d osize=%td\n",

            dc->pc - pc_start, tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf);

    }

#endif

#endif

}
