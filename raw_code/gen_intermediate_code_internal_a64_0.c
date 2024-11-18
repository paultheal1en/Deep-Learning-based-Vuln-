void gen_intermediate_code_internal_a64(ARMCPU *cpu,

                                        TranslationBlock *tb,

                                        bool search_pc)

{

    CPUState *cs = CPU(cpu);

    CPUARMState *env = &cpu->env;

    DisasContext dc1, *dc = &dc1;

    CPUBreakpoint *bp;

    int j, lj;

    target_ulong pc_start;

    target_ulong next_page_start;

    int num_insns;

    int max_insns;



    pc_start = tb->pc;



    dc->tb = tb;



    dc->is_jmp = DISAS_NEXT;

    dc->pc = pc_start;

    dc->singlestep_enabled = cs->singlestep_enabled;

    dc->condjmp = 0;



    dc->aarch64 = 1;

    dc->el3_is_aa64 = arm_el_is_aa64(env, 3);

    dc->thumb = 0;

    dc->bswap_code = 0;

    dc->condexec_mask = 0;

    dc->condexec_cond = 0;

    dc->mmu_idx = ARM_TBFLAG_MMUIDX(tb->flags);

    dc->current_el = arm_mmu_idx_to_el(dc->mmu_idx);

#if !defined(CONFIG_USER_ONLY)

    dc->user = (dc->current_el == 0);

#endif

    dc->cpacr_fpen = ARM_TBFLAG_AA64_FPEN(tb->flags);

    dc->vec_len = 0;

    dc->vec_stride = 0;

    dc->cp_regs = cpu->cp_regs;

    dc->features = env->features;



    /* Single step state. The code-generation logic here is:

     *  SS_ACTIVE == 0:

     *   generate code with no special handling for single-stepping (except

     *   that anything that can make us go to SS_ACTIVE == 1 must end the TB;

     *   this happens anyway because those changes are all system register or

     *   PSTATE writes).

     *  SS_ACTIVE == 1, PSTATE.SS == 1: (active-not-pending)

     *   emit code for one insn

     *   emit code to clear PSTATE.SS

     *   emit code to generate software step exception for completed step

     *   end TB (as usual for having generated an exception)

     *  SS_ACTIVE == 1, PSTATE.SS == 0: (active-pending)

     *   emit code to generate a software step exception

     *   end the TB

     */

    dc->ss_active = ARM_TBFLAG_SS_ACTIVE(tb->flags);

    dc->pstate_ss = ARM_TBFLAG_PSTATE_SS(tb->flags);

    dc->is_ldex = false;

    dc->ss_same_el = (arm_debug_target_el(env) == dc->current_el);



    init_tmp_a64_array(dc);



    next_page_start = (pc_start & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;

    lj = -1;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }



    gen_tb_start(tb);



    tcg_clear_temp_count();



    do {

        if (unlikely(!QTAILQ_EMPTY(&cs->breakpoints))) {

            QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

                if (bp->pc == dc->pc) {

                    gen_exception_internal_insn(dc, 0, EXCP_DEBUG);

                    /* Advance PC so that clearing the breakpoint will

                       invalidate this TB.  */

                    dc->pc += 2;

                    goto done_generating;

                }

            }

        }



        if (search_pc) {

            j = tcg_op_buf_count();

            if (lj < j) {

                lj++;

                while (lj < j) {

                    tcg_ctx.gen_opc_instr_start[lj++] = 0;

                }

            }

            tcg_ctx.gen_opc_pc[lj] = dc->pc;

            tcg_ctx.gen_opc_instr_start[lj] = 1;

            tcg_ctx.gen_opc_icount[lj] = num_insns;

        }



        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO)) {

            gen_io_start();

        }



        if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP | CPU_LOG_TB_OP_OPT))) {

            tcg_gen_debug_insn_start(dc->pc);

        }



        if (dc->ss_active && !dc->pstate_ss) {

            /* Singlestep state is Active-pending.

             * If we're in this state at the start of a TB then either

             *  a) we just took an exception to an EL which is being debugged

             *     and this is the first insn in the exception handler

             *  b) debug exceptions were masked and we just unmasked them

             *     without changing EL (eg by clearing PSTATE.D)

             * In either case we're going to take a swstep exception in the

             * "did not step an insn" case, and so the syndrome ISV and EX

             * bits should be zero.

             */

            assert(num_insns == 0);

            gen_exception(EXCP_UDEF, syn_swstep(dc->ss_same_el, 0, 0),

                          default_exception_el(dc));

            dc->is_jmp = DISAS_EXC;

            break;

        }



        disas_a64_insn(env, dc);



        if (tcg_check_temp_count()) {

            fprintf(stderr, "TCG temporary leak before "TARGET_FMT_lx"\n",

                    dc->pc);

        }



        /* Translation stops when a conditional branch is encountered.

         * Otherwise the subsequent code could get translated several times.

         * Also stop translation when a page boundary is reached.  This

         * ensures prefetch aborts occur at the right place.

         */

        num_insns++;

    } while (!dc->is_jmp && !tcg_op_buf_full() &&

             !cs->singlestep_enabled &&

             !singlestep &&

             !dc->ss_active &&

             dc->pc < next_page_start &&

             num_insns < max_insns);



    if (tb->cflags & CF_LAST_IO) {

        gen_io_end();

    }



    if (unlikely(cs->singlestep_enabled || dc->ss_active)

        && dc->is_jmp != DISAS_EXC) {

        /* Note that this means single stepping WFI doesn't halt the CPU.

         * For conditional branch insns this is harmless unreachable code as

         * gen_goto_tb() has already handled emitting the debug exception

         * (and thus a tb-jump is not possible when singlestepping).

         */

        assert(dc->is_jmp != DISAS_TB_JUMP);

        if (dc->is_jmp != DISAS_JUMP) {

            gen_a64_set_pc_im(dc->pc);

        }

        if (cs->singlestep_enabled) {

            gen_exception_internal(EXCP_DEBUG);

        } else {

            gen_step_complete_exception(dc);

        }

    } else {

        switch (dc->is_jmp) {

        case DISAS_NEXT:

            gen_goto_tb(dc, 1, dc->pc);

            break;

        default:

        case DISAS_UPDATE:

            gen_a64_set_pc_im(dc->pc);

            /* fall through */

        case DISAS_JUMP:

            /* indicate that the hash table must be used to find the next TB */

            tcg_gen_exit_tb(0);

            break;

        case DISAS_TB_JUMP:

        case DISAS_EXC:

        case DISAS_SWI:

            break;

        case DISAS_WFE:

            gen_a64_set_pc_im(dc->pc);

            gen_helper_wfe(cpu_env);

            break;

        case DISAS_WFI:

            /* This is a special case because we don't want to just halt the CPU

             * if trying to debug across a WFI.

             */

            gen_a64_set_pc_im(dc->pc);

            gen_helper_wfi(cpu_env);

            break;

        }

    }



done_generating:

    gen_tb_end(tb, num_insns);



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("----------------\n");

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(env, pc_start, dc->pc - pc_start,

                         4 | (dc->bswap_code << 1));

        qemu_log("\n");

    }

#endif

    if (search_pc) {

        j = tcg_op_buf_count();

        lj++;

        while (lj <= j) {

            tcg_ctx.gen_opc_instr_start[lj++] = 0;

        }

    } else {

        tb->size = dc->pc - pc_start;

        tb->icount = num_insns;

    }

}
