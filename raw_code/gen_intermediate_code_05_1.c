void gen_intermediate_code(CPUAlphaState *env, struct TranslationBlock *tb)
{
    AlphaCPU *cpu = alpha_env_get_cpu(env);
    CPUState *cs = CPU(cpu);
    DisasContext ctx, *ctxp = &ctx;
    target_ulong pc_start;
    target_ulong pc_mask;
    uint32_t insn;
    ExitStatus ret;
    int num_insns;
    int max_insns;
    pc_start = tb->pc;
    ctx.tb = tb;
    ctx.pc = pc_start;
    ctx.tbflags = tb->flags;
    ctx.mem_idx = cpu_mmu_index(env, false);
    ctx.implver = env->implver;
    ctx.amask = env->amask;
    ctx.singlestep_enabled = cs->singlestep_enabled;
#ifdef CONFIG_USER_ONLY
    ctx.ir = cpu_std_ir;
#else
    ctx.palbr = env->palbr;
    ctx.ir = (ctx.tbflags & ENV_FLAG_PAL_MODE ? cpu_pal_ir : cpu_std_ir);
#endif
    /* ??? Every TB begins with unset rounding mode, to be initialized on
       the first fp insn of the TB.  Alternately we could define a proper
       default for every TB (e.g. QUAL_RM_N or QUAL_RM_D) and make sure
       to reset the FP_STATUS to that default at the end of any TB that
       changes the default.  We could even (gasp) dynamiclly figure out
       what default would be most efficient given the running program.  */
    ctx.tb_rm = -1;
    /* Similarly for flush-to-zero.  */
    ctx.tb_ftz = -1;
    TCGV_UNUSED_I64(ctx.zero);
    TCGV_UNUSED_I64(ctx.sink);
    TCGV_UNUSED_I64(ctx.lit);
    num_insns = 0;
    max_insns = tb->cflags & CF_COUNT_MASK;
    if (max_insns == 0) {
        max_insns = CF_COUNT_MASK;
    if (max_insns > TCG_MAX_INSNS) {
        max_insns = TCG_MAX_INSNS;
    if (in_superpage(&ctx, pc_start)) {
        pc_mask = (1ULL << 41) - 1;
    } else {
        pc_mask = ~TARGET_PAGE_MASK;
    gen_tb_start(tb);
    tcg_clear_temp_count();
    do {
        tcg_gen_insn_start(ctx.pc);
        num_insns++;
        if (unlikely(cpu_breakpoint_test(cs, ctx.pc, BP_ANY))) {
            ret = gen_excp(&ctx, EXCP_DEBUG, 0);
            /* The address covered by the breakpoint must be included in
               [tb->pc, tb->pc + tb->size) in order to for it to be
               properly cleared -- thus we increment the PC here so that
               the logic setting tb->size below does the right thing.  */
            ctx.pc += 4;
            break;
        if (num_insns == max_insns && (tb->cflags & CF_LAST_IO)) {
            gen_io_start();
        insn = cpu_ldl_code(env, ctx.pc);
        ctx.pc += 4;
        ret = translate_one(ctxp, insn);
        free_context_temps(ctxp);
        /* If we reach a page boundary, are single stepping,
           or exhaust instruction count, stop generation.  */
        if (ret == NO_EXIT
            && ((ctx.pc & pc_mask) == 0
                || tcg_op_buf_full()
                || num_insns >= max_insns
                || singlestep
                || ctx.singlestep_enabled)) {
            ret = EXIT_FALLTHRU;
    } while (ret == NO_EXIT);
    if (tb->cflags & CF_LAST_IO) {
        gen_io_end();
    switch (ret) {
    case EXIT_GOTO_TB:
    case EXIT_NORETURN:
        break;
    case EXIT_FALLTHRU:
        if (use_goto_tb(&ctx, ctx.pc)) {
            tcg_gen_goto_tb(0);
            tcg_gen_movi_i64(cpu_pc, ctx.pc);
            tcg_gen_exit_tb((uintptr_t)ctx.tb);
        /* FALLTHRU */
    case EXIT_PC_STALE:
        tcg_gen_movi_i64(cpu_pc, ctx.pc);
        /* FALLTHRU */
    case EXIT_PC_UPDATED:
        if (!use_exit_tb(&ctx)) {
            tcg_gen_lookup_and_goto_ptr(cpu_pc);
            break;
        /* FALLTHRU */
    case EXIT_PC_UPDATED_NOCHAIN:
        if (ctx.singlestep_enabled) {
            gen_excp_1(EXCP_DEBUG, 0);
        } else {
            tcg_gen_exit_tb(0);
        break;
    default:
        g_assert_not_reached();
    gen_tb_end(tb, num_insns);
    tb->size = ctx.pc - pc_start;
    tb->icount = num_insns;
#ifdef DEBUG_DISAS
    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)
        && qemu_log_in_addr_range(pc_start)) {
        qemu_log_lock();
        qemu_log("IN: %s\n", lookup_symbol(pc_start));
        log_target_disas(cs, pc_start, ctx.pc - pc_start, 1);
        qemu_log("\n");
        qemu_log_unlock();
#endif