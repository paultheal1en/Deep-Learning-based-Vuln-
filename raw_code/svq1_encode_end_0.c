static av_cold int svq1_encode_end(AVCodecContext *avctx)

{

    SVQ1EncContext *const s = avctx->priv_data;

    int i;



    av_log(avctx, AV_LOG_DEBUG, "RD: %f\n",

           s->rd_total / (double)(avctx->width * avctx->height *

                                  avctx->frame_number));



    s->m.mb_type = NULL;

    ff_mpv_common_end(&s->m);



    av_freep(&s->m.me.scratchpad);

    av_freep(&s->m.me.map);

    av_freep(&s->m.me.score_map);

    av_freep(&s->mb_type);

    av_freep(&s->dummy);

    av_freep(&s->scratchbuf);



    for (i = 0; i < 3; i++) {

        av_freep(&s->motion_val8[i]);

        av_freep(&s->motion_val16[i]);

    }



    av_frame_free(&s->current_picture);

    av_frame_free(&s->last_picture);

    av_frame_free(&avctx->coded_frame);



    return 0;

}
