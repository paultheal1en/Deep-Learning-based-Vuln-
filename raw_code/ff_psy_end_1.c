av_cold void ff_psy_end(FFPsyContext *ctx)

{

    if (ctx->model->end)

        ctx->model->end(ctx);

    av_freep(&ctx->bands);

    av_freep(&ctx->num_bands);

    av_freep(&ctx->group);

    av_freep(&ctx->ch);

}
