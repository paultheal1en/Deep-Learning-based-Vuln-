static int finish_frame(AVCodecContext *avctx, AVFrame *pict)

{

    RV34DecContext *r = avctx->priv_data;

    MpegEncContext *s = &r->s;

    int got_picture = 0;



    ff_er_frame_end(s);

    ff_MPV_frame_end(s);




    if (HAVE_THREADS && (s->avctx->active_thread_type & FF_THREAD_FRAME))

        ff_thread_report_progress(&s->current_picture_ptr->f, INT_MAX, 0);



    if (s->pict_type == AV_PICTURE_TYPE_B || s->low_delay) {

        *pict = s->current_picture_ptr->f;

        got_picture = 1;

    } else if (s->last_picture_ptr != NULL) {

        *pict = s->last_picture_ptr->f;

        got_picture = 1;

    }

    if (got_picture)

        ff_print_debug_info(s, pict);



    return got_picture;

}