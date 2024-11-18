gen_intermediate_code_internal(MicroBlazeCPU *cpu, TranslationBlock *tb,

                               bool search_pc)

{

    CPUState *cs = CPU(cpu);

    CPUMBState *env = &cpu->env;

    uint32_t pc_start;

    int j, lj;

    struct DisasContext ctx;

    struct DisasContext *dc = &ctx;

    uint32_t next_page_start, org_flags;

    target_ulong npc;

    int num_insns;

    int max_insns;



    pc_start = tb->pc;

    dc->cpu = cpu;

    dc->tb = tb;

    org_flags = dc->synced_flags = dc->tb_flags = tb->flags;



    dc->is_jmp = DISAS_NEXT;

    dc->jmp = 0;

    dc->delayed_branch = !!(dc->tb_flags & D_FLAG);

    if (dc->delayed_branch) {

        dc->jmp = JMP_INDIRECT;

    }

    dc->pc = pc_start;

    dc->singlestep_enabled = cs->singlestep_enabled;

    dc->cpustate_changed = 0;

    dc->abort_at_next_insn = 0;

    dc->nr_nops = 0;



    if (pc_start & 3) {

        cpu_abort(cs, "Microblaze: unaligned PC=%x\n", pc_start);

    }



    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

#if !SIM_COMPAT

        qemu_log("--------------\n");

        log_cpu_state(CPU(cpu), 0);

#endif

    }



    next_page_start = (pc_start & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;

    lj = -1;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0)

        max_insns = CF_COUNT_MASK;



    gen_tb_start(tb);

    do

    {

#if SIM_COMPAT

        if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

            tcg_gen_movi_tl(cpu_SR[SR_PC], dc->pc);

            gen_helper_debug();

        }

#endif

        check_breakpoint(env, dc);



        if (search_pc) {

            j = tcg_op_buf_count();

            if (lj < j) {

                lj++;

                while (lj < j)

                    tcg_ctx.gen_opc_instr_start[lj++] = 0;

            }

            tcg_ctx.gen_opc_pc[lj] = dc->pc;

            tcg_ctx.gen_opc_instr_start[lj] = 1;

                        tcg_ctx.gen_opc_icount[lj] = num_insns;

        }



        /* Pretty disas.  */

        LOG_DIS("%8.8x:\t", dc->pc);



        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO))

            gen_io_start();



        dc->clear_imm = 1;

        decode(dc, cpu_ldl_code(env, dc->pc));

        if (dc->clear_imm)

            dc->tb_flags &= ~IMM_FLAG;

        dc->pc += 4;

        num_insns++;



        if (dc->delayed_branch) {

            dc->delayed_branch--;

            if (!dc->delayed_branch) {

                if (dc->tb_flags & DRTI_FLAG)

                    do_rti(dc);

                 if (dc->tb_flags & DRTB_FLAG)

                    do_rtb(dc);

                if (dc->tb_flags & DRTE_FLAG)

                    do_rte(dc);

                /* Clear the delay slot flag.  */

                dc->tb_flags &= ~D_FLAG;

                /* If it is a direct jump, try direct chaining.  */

                if (dc->jmp == JMP_INDIRECT) {

                    eval_cond_jmp(dc, env_btarget, tcg_const_tl(dc->pc));

                    dc->is_jmp = DISAS_JUMP;

                } else if (dc->jmp == JMP_DIRECT) {

                    t_sync_flags(dc);

                    gen_goto_tb(dc, 0, dc->jmp_pc);

                    dc->is_jmp = DISAS_TB_JUMP;

                } else if (dc->jmp == JMP_DIRECT_CC) {

                    int l1;



                    t_sync_flags(dc);

                    l1 = gen_new_label();

                    /* Conditional jmp.  */

                    tcg_gen_brcondi_tl(TCG_COND_NE, env_btaken, 0, l1);

                    gen_goto_tb(dc, 1, dc->pc);

                    gen_set_label(l1);

                    gen_goto_tb(dc, 0, dc->jmp_pc);



                    dc->is_jmp = DISAS_TB_JUMP;

                }

                break;

            }

        }

        if (cs->singlestep_enabled) {

            break;

        }

    } while (!dc->is_jmp && !dc->cpustate_changed

             && !tcg_op_buf_full()

             && !singlestep

             && (dc->pc < next_page_start)

             && num_insns < max_insns);



    npc = dc->pc;

    if (dc->jmp == JMP_DIRECT || dc->jmp == JMP_DIRECT_CC) {

        if (dc->tb_flags & D_FLAG) {

            dc->is_jmp = DISAS_UPDATE;

            tcg_gen_movi_tl(cpu_SR[SR_PC], npc);

            sync_jmpstate(dc);

        } else

            npc = dc->jmp_pc;

    }



    if (tb->cflags & CF_LAST_IO)

        gen_io_end();

    /* Force an update if the per-tb cpu state has changed.  */

    if (dc->is_jmp == DISAS_NEXT

        && (dc->cpustate_changed || org_flags != dc->tb_flags)) {

        dc->is_jmp = DISAS_UPDATE;

        tcg_gen_movi_tl(cpu_SR[SR_PC], npc);

    }

    t_sync_flags(dc);



    if (unlikely(cs->singlestep_enabled)) {

        TCGv_i32 tmp = tcg_const_i32(EXCP_DEBUG);



        if (dc->is_jmp != DISAS_JUMP) {

            tcg_gen_movi_tl(cpu_SR[SR_PC], npc);

        }

        gen_helper_raise_exception(cpu_env, tmp);

        tcg_temp_free_i32(tmp);

    } else {

        switch(dc->is_jmp) {

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

            case DISAS_TB_JUMP:

                /* nothing more to generate */

                break;

        }

    }

    gen_tb_end(tb, num_insns);



    if (search_pc) {

        j = tcg_op_buf_count();

        lj++;

        while (lj <= j)

            tcg_ctx.gen_opc_instr_start[lj++] = 0;

    } else {

        tb->size = dc->pc - pc_start;

                tb->icount = num_insns;

    }



#ifdef DEBUG_DISAS

#if !SIM_COMPAT

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("\n");

#if DISAS_GNU

        log_target_disas(env, pc_start, dc->pc - pc_start, 0);

#endif

        qemu_log("\nisize=%d osize=%d\n",

                 dc->pc - pc_start, tcg_op_buf_count());

    }

#endif

#endif

    assert(!dc->abort_at_next_insn);

}
