static void gen_cli(DisasContext *ctx)

{

    /* Cache line invalidate: privileged and treated as no-op */

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

#endif

}
