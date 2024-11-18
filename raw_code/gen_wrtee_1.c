static void gen_wrtee(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    TCGv t0;

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    t0 = tcg_temp_new();

    tcg_gen_andi_tl(t0, cpu_gpr[rD(ctx->opcode)], (1 << MSR_EE));

    tcg_gen_andi_tl(cpu_msr, cpu_msr, ~(1 << MSR_EE));

    tcg_gen_or_tl(cpu_msr, cpu_msr, t0);

    tcg_temp_free(t0);

    /* Stop translation to have a chance to raise an exception

     * if we just set msr_ee to 1

     */

    gen_stop_exception(ctx);

#endif

}
