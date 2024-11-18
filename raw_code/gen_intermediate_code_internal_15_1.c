static always_inline void gen_intermediate_code_internal (CPUState *env,

                                                          TranslationBlock *tb,

                                                          int search_pc)

{

#if defined ALPHA_DEBUG_DISAS

    static int insn_count;

#endif

    DisasContext ctx, *ctxp = &ctx;

    target_ulong pc_start;

    uint32_t insn;

    uint16_t *gen_opc_end;

    CPUBreakpoint *bp;

    int j, lj = -1;

    int ret;

    int num_insns;

    int max_insns;



    pc_start = tb->pc;

    gen_opc_end = gen_opc_buf + OPC_MAX_SIZE;

    ctx.pc = pc_start;

    ctx.amask = env->amask;

#if defined (CONFIG_USER_ONLY)

    ctx.mem_idx = 0;

#else

    ctx.mem_idx = ((env->ps >> 3) & 3);

    ctx.pal_mode = env->ipr[IPR_EXC_ADDR] & 1;

#endif

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0)

        max_insns = CF_COUNT_MASK;



    gen_icount_start();

    for (ret = 0; ret == 0;) {

        if (unlikely(!TAILQ_EMPTY(&env->breakpoints))) {

            TAILQ_FOREACH(bp, &env->breakpoints, entry) {

                if (bp->pc == ctx.pc) {

                    gen_excp(&ctx, EXCP_DEBUG, 0);

                    break;

                }

            }

        }

        if (search_pc) {

            j = gen_opc_ptr - gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j)

                    gen_opc_instr_start[lj++] = 0;

                gen_opc_pc[lj] = ctx.pc;

                gen_opc_instr_start[lj] = 1;

                gen_opc_icount[lj] = num_insns;

            }

        }

        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO))

            gen_io_start();

#if defined ALPHA_DEBUG_DISAS

        insn_count++;

        LOG_DISAS("pc " TARGET_FMT_lx " mem_idx %d\n",

                  ctx.pc, ctx.mem_idx);

#endif

        insn = ldl_code(ctx.pc);

#if defined ALPHA_DEBUG_DISAS

        insn_count++;

        LOG_DISAS("opcode %08x %d\n", insn, insn_count);

#endif

        num_insns++;

        ctx.pc += 4;

        ret = translate_one(ctxp, insn);

        if (ret != 0)

            break;

        /* if we reach a page boundary or are single stepping, stop

         * generation

         */

        if (((ctx.pc & (TARGET_PAGE_SIZE - 1)) == 0) ||

            num_insns >= max_insns) {

            break;

        }



        if (env->singlestep_enabled) {

            gen_excp(&ctx, EXCP_DEBUG, 0);

            break;

	}



#if defined (DO_SINGLE_STEP)

        break;

#endif

    }

    if (ret != 1 && ret != 3) {

        tcg_gen_movi_i64(cpu_pc, ctx.pc);

    }

#if defined (DO_TB_FLUSH)

    gen_helper_tb_flush();

#endif

    if (tb->cflags & CF_LAST_IO)

        gen_io_end();

    /* Generate the return instruction */

    tcg_gen_exit_tb(0);

    gen_icount_end(tb, num_insns);

    *gen_opc_ptr = INDEX_op_end;

    if (search_pc) {

        j = gen_opc_ptr - gen_opc_buf;

        lj++;

        while (lj <= j)

            gen_opc_instr_start[lj++] = 0;

    } else {

        tb->size = ctx.pc - pc_start;

        tb->icount = num_insns;

    }

#if defined ALPHA_DEBUG_DISAS

    log_cpu_state_mask(CPU_LOG_TB_CPU, env, 0);

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(pc_start, ctx.pc - pc_start, 1);

        qemu_log("\n");

    }

#endif

}
