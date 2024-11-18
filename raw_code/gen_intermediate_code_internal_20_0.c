gen_intermediate_code_internal(TriCoreCPU *cpu, struct TranslationBlock *tb,

                              int search_pc)

{

    CPUState *cs = CPU(cpu);

    CPUTriCoreState *env = &cpu->env;

    DisasContext ctx;

    target_ulong pc_start;

    int num_insns;

    uint16_t *gen_opc_end;



    if (search_pc) {

        qemu_log("search pc %d\n", search_pc);

    }



    num_insns = 0;

    pc_start = tb->pc;

    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;

    ctx.pc = pc_start;

    ctx.saved_pc = -1;

    ctx.tb = tb;

    ctx.singlestep_enabled = cs->singlestep_enabled;

    ctx.bstate = BS_NONE;

    ctx.mem_idx = cpu_mmu_index(env);



    tcg_clear_temp_count();

    gen_tb_start();

    while (ctx.bstate == BS_NONE) {

        ctx.opcode = cpu_ldl_code(env, ctx.pc);

        decode_opc(env, &ctx, 0);



        num_insns++;



        if (tcg_ctx.gen_opc_ptr >= gen_opc_end) {

            gen_save_pc(ctx.next_pc);

            tcg_gen_exit_tb(0);

            break;

        }

        if (singlestep) {

            gen_save_pc(ctx.next_pc);

            tcg_gen_exit_tb(0);

            break;

        }

        ctx.pc = ctx.next_pc;

    }



    gen_tb_end(tb, num_insns);

    *tcg_ctx.gen_opc_ptr = INDEX_op_end;

    if (search_pc) {

        printf("done_generating search pc\n");

    } else {

        tb->size = ctx.pc - pc_start;

        tb->icount = num_insns;

    }

    if (tcg_check_temp_count()) {

        printf("LEAK at %08x\n", env->PC);

    }



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(env, pc_start, ctx.pc - pc_start, 0);

        qemu_log("\n");

    }

#endif

}
