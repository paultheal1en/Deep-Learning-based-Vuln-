static int finish_frame(AVCodecContext *avctx, AVFrame *pict)

{

    RV34DecContext *r = avctx->priv_data;

    MpegEncContext *s = &r->s;

    int got_picture = 0, ret;



    ff_er_frame_end(&s->er);

    ff_mpv_frame_end(s);

    s->mb_num_left = 0;



    if (HAVE_THREADS && (s->avctx->active_thread_type & FF_THREAD_FRAME))

        ff_thread_report_progress(&s->current_picture_ptr->tf, INT_MAX, 0);



    if (s->pict_type == AV_PICTURE_TYPE_B || s->low_delay) {

        if ((ret = av_frame_ref(pict, s->current_picture_ptr->f)) < 0)

            return ret;

        ff_print_debug_info(s, s->current_picture_ptr);

        got_picture = 1;

    } else if (s->last_picture_ptr != NULL) {

        if ((ret = av_frame_ref(pict, s->last_picture_ptr->f)) < 0)

            return ret;

        ff_print_debug_info(s, s->last_picture_ptr);

        got_picture = 1;

    }



    return got_picture;

}
