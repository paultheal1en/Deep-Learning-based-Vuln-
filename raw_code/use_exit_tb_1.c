static bool use_exit_tb(DisasContext *ctx)

{

    return ((ctx->base.tb->cflags & CF_LAST_IO)

            || ctx->base.singlestep_enabled

            || singlestep);

}
