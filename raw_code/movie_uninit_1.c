static av_cold void movie_uninit(AVFilterContext *ctx)

{

    MovieContext *movie = ctx->priv;

    int i;



    for (i = 0; i < ctx->nb_outputs; i++) {

        av_freep(&ctx->output_pads[i].name);

        if (movie->st[i].st)

            avcodec_close(movie->st[i].st->codec);

    }

    av_freep(&movie->st);

    av_freep(&movie->out_index);

    av_frame_free(&movie->frame);

    if (movie->format_ctx)

        avformat_close_input(&movie->format_ctx);

}
