void gen_intermediate_code(CPUState *cs, TranslationBlock * tb)

{

    CPUSPARCState *env = cs->env_ptr;

    target_ulong pc_start, last_pc;

    DisasContext dc1, *dc = &dc1;

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

    dc->mem_idx = tb->flags & TB_FLAG_MMU_MASK;

    dc->def = &env->def;

    dc->fpu_enabled = tb_fpu_enabled(tb->flags);

    dc->address_mask_32bit = tb_am_enabled(tb->flags);

    dc->singlestep = (cs->singlestep_enabled || singlestep);

#ifndef CONFIG_USER_ONLY

    dc->supervisor = (tb->flags & TB_FLAG_SUPER) != 0;

#endif

#ifdef TARGET_SPARC64

    dc->fprs_dirty = 0;

    dc->asi = (tb->flags >> TB_FLAG_ASI_SHIFT) & 0xff;

#ifndef CONFIG_USER_ONLY

    dc->hypervisor = (tb->flags & TB_FLAG_HYPER) != 0;

#endif

#endif



    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }

    if (max_insns > TCG_MAX_INSNS) {

        max_insns = TCG_MAX_INSNS;

    }



    gen_tb_start(tb);

    do {

        if (dc->npc & JUMP_PC) {

            assert(dc->jump_pc[1] == dc->pc + 4);

            tcg_gen_insn_start(dc->pc, dc->jump_pc[0] | JUMP_PC);

        } else {

            tcg_gen_insn_start(dc->pc, dc->npc);

        }

        num_insns++;

        last_pc = dc->pc;



        if (unlikely(cpu_breakpoint_test(cs, dc->pc, BP_ANY))) {

            if (dc->pc != pc_start) {

                save_state(dc);

            }

            gen_helper_debug(cpu_env);

            tcg_gen_exit_tb(0);

            dc->is_br = 1;

            goto exit_gen_loop;

        }



        if (num_insns == max_insns && (tb->cflags & CF_LAST_IO)) {

            gen_io_start();

        }



        insn = cpu_ldl_code(env, dc->pc);



        disas_sparc_insn(dc, insn);



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

    } while (!tcg_op_buf_full() &&

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



    tb->size = last_pc + 4 - pc_start;

    tb->icount = num_insns;



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)

        && qemu_log_in_addr_range(pc_start)) {

        qemu_log_lock();

        qemu_log("--------------\n");

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(cs, pc_start, last_pc + 4 - pc_start, 0);

        qemu_log("\n");

        qemu_log_unlock();

    }

#endif

}
