static int teletext_decode_frame(AVCodecContext *avctx, void *data, int *data_size, AVPacket *pkt)

{

    TeletextContext *ctx = avctx->priv_data;

    AVSubtitle      *sub = data;

    int             ret = 0;



    if (!ctx->vbi) {

        if (!(ctx->vbi = vbi_decoder_new()))

            return AVERROR(ENOMEM);

        if (!vbi_event_handler_add(ctx->vbi, VBI_EVENT_TTX_PAGE, handler, ctx)) {

            vbi_decoder_delete(ctx->vbi);

            ctx->vbi = NULL;

            return AVERROR(ENOMEM);

        }

    }



    if (avctx->pkt_timebase.den && pkt->pts != AV_NOPTS_VALUE)

        ctx->pts = av_rescale_q(pkt->pts, avctx->pkt_timebase, AV_TIME_BASE_Q);



    if (pkt->size) {

        int lines;

        const int full_pes_size = pkt->size + 45; /* PES header is 45 bytes */



        // We allow unreasonably big packets, even if the standard only allows a max size of 1472

        if (full_pes_size < 184 || full_pes_size > 65504 || full_pes_size % 184 != 0)

            return AVERROR_INVALIDDATA;



        ctx->handler_ret = pkt->size;



        if (data_identifier_is_teletext(*pkt->data)) {

            if ((lines = slice_to_vbi_lines(ctx, pkt->data + 1, pkt->size - 1)) < 0)

                return lines;

            av_dlog(avctx, "ctx=%p buf_size=%d lines=%u pkt_pts=%7.3f\n",

                    ctx, pkt->size, lines, (double)pkt->pts/90000.0);

            if (lines > 0) {

#ifdef DEBUG

                int i;

                av_log(avctx, AV_LOG_DEBUG, "line numbers:");

                for(i = 0; i < lines; i++)

                    av_log(avctx, AV_LOG_DEBUG, " %d", ctx->sliced[i].line);

                av_log(avctx, AV_LOG_DEBUG, "\n");

#endif

                vbi_decode(ctx->vbi, ctx->sliced, lines, 0.0);

                ctx->lines_processed += lines;

            }

        }

        ctx->pts = AV_NOPTS_VALUE;

        ret = ctx->handler_ret;

    }



    if (ret < 0)

        return ret;



    // is there a subtitle to pass?

    if (ctx->nb_pages) {

        int i;

        sub->format = ctx->format_id;

        sub->start_display_time = 0;

        sub->end_display_time = ctx->sub_duration;

        sub->num_rects = 0;

        sub->pts = ctx->pages->pts;



        if (ctx->pages->sub_rect->type != SUBTITLE_NONE) {

            sub->rects = av_malloc(sizeof(*sub->rects));

            if (sub->rects) {

                sub->num_rects = 1;

                sub->rects[0] = ctx->pages->sub_rect;

            } else {

                ret = AVERROR(ENOMEM);

            }

        } else {

            av_log(avctx, AV_LOG_DEBUG, "sending empty sub\n");

            sub->rects = NULL;

        }

        if (!sub->rects) // no rect was passed

            subtitle_rect_free(&ctx->pages->sub_rect);



        for (i = 0; i < ctx->nb_pages - 1; i++)

            ctx->pages[i] = ctx->pages[i + 1];

        ctx->nb_pages--;



        if (ret >= 0)

            *data_size = 1;

    } else

        *data_size = 0;



    return ret;

}
