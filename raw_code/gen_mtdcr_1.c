static void gen_mtdcr(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

#else

    TCGv dcrn;

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

        return;

    }

    /* NIP cannot be restored if the memory exception comes from an helper */

    gen_update_nip(ctx, ctx->nip - 4);

    dcrn = tcg_const_tl(SPR(ctx->opcode));

    gen_helper_store_dcr(cpu_env, dcrn, cpu_gpr[rS(ctx->opcode)]);

    tcg_temp_free(dcrn);

#endif

}
