static void gen_tlbie(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    if (unlikely(ctx->pr || !ctx->hv)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    if (NARROW_MODE(ctx)) {

        TCGv t0 = tcg_temp_new();

        tcg_gen_ext32u_tl(t0, cpu_gpr[rB(ctx->opcode)]);

        gen_helper_tlbie(cpu_env, t0);

        tcg_temp_free(t0);

    } else {

        gen_helper_tlbie(cpu_env, cpu_gpr[rB(ctx->opcode)]);

    }

#endif

}
