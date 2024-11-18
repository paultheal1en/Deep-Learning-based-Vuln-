static inline void RET_STOP (DisasContext *ctx)

{

    gen_update_nip(ctx, ctx->nip);

    ctx->exception = EXCP_MTMSR;

}
