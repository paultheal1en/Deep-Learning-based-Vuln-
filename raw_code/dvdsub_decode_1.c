static int dvdsub_decode(AVCodecContext *avctx,

                         void *data, int *data_size,

                         AVPacket *avpkt)

{

    DVDSubContext *ctx = avctx->priv_data;

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    AVSubtitle *sub = data;

    int is_menu;



    if (ctx->buf) {

        int ret = append_to_cached_buf(avctx, buf, buf_size);

        if (ret < 0) {

            *data_size = 0;

            return ret;

        }

        buf = ctx->buf;

        buf_size = ctx->buf_size;

    }



    is_menu = decode_dvd_subtitles(ctx, sub, buf, buf_size);

    if (is_menu == AVERROR(EAGAIN)) {

        *data_size = 0;

        return append_to_cached_buf(avctx, buf, buf_size);

    }



    if (is_menu < 0) {

    no_subtitle:

        reset_rects(sub);

        *data_size = 0;



        return buf_size;

    }

    if (!is_menu && find_smallest_bounding_rectangle(sub) == 0)

        goto no_subtitle;



    if (ctx->forced_subs_only && !(sub->rects[0]->flags & AV_SUBTITLE_FLAG_FORCED))

        goto no_subtitle;



#if defined(DEBUG)

    {

    char ppm_name[32];



    snprintf(ppm_name, sizeof(ppm_name), "/tmp/%05d.ppm", ctx->sub_id++);

    av_dlog(NULL, "start=%d ms end =%d ms\n",

            sub->start_display_time,

            sub->end_display_time);

    ppm_save(ppm_name, sub->rects[0]->pict.data[0],

             sub->rects[0]->w, sub->rects[0]->h, (uint32_t*) sub->rects[0]->pict.data[1]);

    }

#endif



    av_freep(&ctx->buf);

    ctx->buf_size = 0;

    *data_size = 1;

    return buf_size;

}
