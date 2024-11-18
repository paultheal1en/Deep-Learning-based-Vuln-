static void gen_hrfid(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    /* Restore CPU state */

    if (unlikely(ctx->pr || !ctx->hv)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    gen_helper_hrfid(cpu_env);

    gen_sync_exception(ctx);

#endif

}
