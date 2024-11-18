gen_intermediate_code_internal (CPUState *env, TranslationBlock *tb,

                                int search_pc)

{

    DisasContext ctx;

    target_ulong pc_start;

    uint16_t *gen_opc_end;

    int j, lj = -1;



    if (search_pc && loglevel)

        fprintf (logfile, "search pc %d\n", search_pc);



    pc_start = tb->pc;

    gen_opc_end = gen_opc_buf + OPC_MAX_SIZE;

    ctx.pc = pc_start;

    ctx.saved_pc = -1;

    ctx.tb = tb;

    ctx.bstate = BS_NONE;

    /* Restore delay slot state from the tb context.  */

    ctx.hflags = (uint32_t)tb->flags; /* FIXME: maybe use 64 bits here? */

    restore_cpu_state(env, &ctx);

#if defined(CONFIG_USER_ONLY)

    ctx.mem_idx = MIPS_HFLAG_UM;

#else

    ctx.mem_idx = ctx.hflags & MIPS_HFLAG_KSU;

#endif

#ifdef DEBUG_DISAS

    if (loglevel & CPU_LOG_TB_CPU) {

        fprintf(logfile, "------------------------------------------------\n");

        /* FIXME: This may print out stale hflags from env... */

        cpu_dump_state(env, logfile, fprintf, 0);

    }

#endif

#ifdef MIPS_DEBUG_DISAS

    if (loglevel & CPU_LOG_TB_IN_ASM)

        fprintf(logfile, "\ntb %p idx %d hflags %04x\n",

                tb, ctx.mem_idx, ctx.hflags);

#endif

    while (ctx.bstate == BS_NONE && gen_opc_ptr < gen_opc_end) {

        if (env->nb_breakpoints > 0) {

            for(j = 0; j < env->nb_breakpoints; j++) {

                if (env->breakpoints[j] == ctx.pc) {

                    save_cpu_state(&ctx, 1);

                    ctx.bstate = BS_BRANCH;

                    gen_op_debug();

                    /* Include the breakpoint location or the tb won't

                     * be flushed when it must be.  */

                    ctx.pc += 4;

                    goto done_generating;

                }

            }

        }



        if (search_pc) {

            j = gen_opc_ptr - gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j)

                    gen_opc_instr_start[lj++] = 0;

            }

            gen_opc_pc[lj] = ctx.pc;

            gen_opc_hflags[lj] = ctx.hflags & MIPS_HFLAG_BMASK;

            gen_opc_instr_start[lj] = 1;

        }

        ctx.opcode = ldl_code(ctx.pc);

        decode_opc(env, &ctx);

        ctx.pc += 4;



        if (env->singlestep_enabled)

            break;



        if ((ctx.pc & (TARGET_PAGE_SIZE - 1)) == 0)

            break;



#if defined (MIPS_SINGLE_STEP)

        break;

#endif

    }

    if (env->singlestep_enabled) {

        save_cpu_state(&ctx, ctx.bstate == BS_NONE);

        gen_op_debug();

    } else {

	switch (ctx.bstate) {

        case BS_STOP:

            tcg_gen_helper_0_0(do_interrupt_restart);

            gen_goto_tb(&ctx, 0, ctx.pc);

            break;

        case BS_NONE:

            save_cpu_state(&ctx, 0);

            gen_goto_tb(&ctx, 0, ctx.pc);

            break;

        case BS_EXCP:

            tcg_gen_helper_0_0(do_interrupt_restart);

            tcg_gen_exit_tb(0);

            break;

        case BS_BRANCH:

        default:

            break;

	}

    }

done_generating:

    *gen_opc_ptr = INDEX_op_end;

    if (search_pc) {

        j = gen_opc_ptr - gen_opc_buf;

        lj++;

        while (lj <= j)

            gen_opc_instr_start[lj++] = 0;

    } else {

        tb->size = ctx.pc - pc_start;

    }

#ifdef DEBUG_DISAS

#if defined MIPS_DEBUG_DISAS

    if (loglevel & CPU_LOG_TB_IN_ASM)

        fprintf(logfile, "\n");

#endif

    if (loglevel & CPU_LOG_TB_IN_ASM) {

        fprintf(logfile, "IN: %s\n", lookup_symbol(pc_start));

        target_disas(logfile, pc_start, ctx.pc - pc_start, 0);

        fprintf(logfile, "\n");

    }

    if (loglevel & CPU_LOG_TB_CPU) {

        fprintf(logfile, "---------------- %d %08x\n", ctx.bstate, ctx.hflags);

    }

#endif



    return 0;

}
