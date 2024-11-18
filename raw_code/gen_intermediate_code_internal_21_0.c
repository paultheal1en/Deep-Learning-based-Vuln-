gen_intermediate_code_internal(MoxieCPU *cpu, TranslationBlock *tb,

                               bool search_pc)

{

    CPUState *cs = CPU(cpu);

    DisasContext ctx;

    target_ulong pc_start;

    uint16_t *gen_opc_end;

    CPUBreakpoint *bp;

    int j, lj = -1;

    CPUMoxieState *env = &cpu->env;

    int num_insns;



    pc_start = tb->pc;

    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;

    ctx.pc = pc_start;

    ctx.saved_pc = -1;

    ctx.tb = tb;

    ctx.memidx = 0;

    ctx.singlestep_enabled = 0;

    ctx.bstate = BS_NONE;

    num_insns = 0;



    gen_tb_start();

    do {

        if (unlikely(!QTAILQ_EMPTY(&cs->breakpoints))) {

            QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

                if (ctx.pc == bp->pc) {

                    tcg_gen_movi_i32(cpu_pc, ctx.pc);

                    gen_helper_debug(cpu_env);

                    ctx.bstate = BS_EXCP;

                    goto done_generating;

                }

            }

        }



        if (search_pc) {

            j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j) {

                    tcg_ctx.gen_opc_instr_start[lj++] = 0;

                }

            }

            tcg_ctx.gen_opc_pc[lj] = ctx.pc;

            tcg_ctx.gen_opc_instr_start[lj] = 1;

            tcg_ctx.gen_opc_icount[lj] = num_insns;

        }

        ctx.opcode = cpu_lduw_code(env, ctx.pc);

        ctx.pc += decode_opc(cpu, &ctx);

        num_insns++;



        if (cs->singlestep_enabled) {

            break;

        }



        if ((ctx.pc & (TARGET_PAGE_SIZE - 1)) == 0) {

            break;

        }

    } while (ctx.bstate == BS_NONE && tcg_ctx.gen_opc_ptr < gen_opc_end);



    if (cs->singlestep_enabled) {

        tcg_gen_movi_tl(cpu_pc, ctx.pc);

        gen_helper_debug(cpu_env);

    } else {

        switch (ctx.bstate) {

        case BS_STOP:

        case BS_NONE:

            gen_goto_tb(env, &ctx, 0, ctx.pc);

            break;

        case BS_EXCP:

            tcg_gen_exit_tb(0);

            break;

        case BS_BRANCH:

        default:

            break;

        }

    }

 done_generating:

    gen_tb_end(tb, num_insns);

    *tcg_ctx.gen_opc_ptr = INDEX_op_end;

    if (search_pc) {

        j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

        lj++;

        while (lj <= j) {

            tcg_ctx.gen_opc_instr_start[lj++] = 0;

        }

    } else {

        tb->size = ctx.pc - pc_start;

        tb->icount = num_insns;

    }

}
