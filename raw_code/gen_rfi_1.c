static void gen_rfi(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    /* FIXME: This instruction doesn't exist anymore on 64-bit server

     * processors compliant with arch 2.x, we should remove it there,

     * but we need to fix OpenBIOS not to use it on 970 first

     */

    /* Restore CPU state */

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    gen_update_cfar(ctx, ctx->nip);

    gen_helper_rfi(cpu_env);

    gen_sync_exception(ctx);

#endif

}
