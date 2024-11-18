static av_cold int vp9_decode_free(AVCodecContext *ctx)

{

    VP9Context *s = ctx->priv_data;

    int i;



    for (i = 0; i < 2; i++) {

        if (s->frames[i].tf.f->data[0])

            vp9_unref_frame(ctx, &s->frames[i]);

        av_frame_free(&s->frames[i].tf.f);

    }

    for (i = 0; i < 8; i++) {

        if (s->refs[i].f->data[0])

            ff_thread_release_buffer(ctx, &s->refs[i]);

        av_frame_free(&s->refs[i].f);

        if (s->next_refs[i].f->data[0])

            ff_thread_release_buffer(ctx, &s->next_refs[i]);

        av_frame_free(&s->next_refs[i].f);

    }

    av_freep(&s->above_partition_ctx);

    av_freep(&s->c_b);

    s->c_b_size = 0;

    av_freep(&s->b_base);

    av_freep(&s->block_base);



    return 0;

}
