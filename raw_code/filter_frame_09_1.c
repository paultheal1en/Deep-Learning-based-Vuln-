static int filter_frame(AVFilterLink *inlink, AVFrame *src_buffer)

{

    AVFilterContext  *ctx = inlink->dst;

    ATempoContext *atempo = ctx->priv;

    AVFilterLink *outlink = ctx->outputs[0];



    int ret = 0;

    int n_in = src_buffer->nb_samples;

    int n_out = (int)(0.5 + ((double)n_in) / atempo->tempo);



    const uint8_t *src = src_buffer->data[0];

    const uint8_t *src_end = src + n_in * atempo->stride;



    while (src < src_end) {

        if (!atempo->dst_buffer) {

            atempo->dst_buffer = ff_get_audio_buffer(outlink, n_out);

            if (!atempo->dst_buffer)

                return AVERROR(ENOMEM);

            av_frame_copy_props(atempo->dst_buffer, src_buffer);



            atempo->dst = atempo->dst_buffer->data[0];

            atempo->dst_end = atempo->dst + n_out * atempo->stride;

        }



        yae_apply(atempo, &src, src_end, &atempo->dst, atempo->dst_end);



        if (atempo->dst == atempo->dst_end) {

            int n_samples = ((atempo->dst - atempo->dst_buffer->data[0]) /

                             atempo->stride);

            ret = push_samples(atempo, outlink, n_samples);

            if (ret < 0)

                goto end;

        }

    }



    atempo->nsamples_in += n_in;

end:

    av_frame_free(&src_buffer);

    return ret;

}
