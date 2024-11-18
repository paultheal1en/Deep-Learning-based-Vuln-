static void vp3_draw_horiz_band(Vp3DecodeContext *s, int y)

{

    int h, cy, i;

    int offset[AV_NUM_DATA_POINTERS];



    if (HAVE_THREADS && s->avctx->active_thread_type & FF_THREAD_FRAME) {

        int y_flipped = s->flipped_image ? s->avctx->height - y : y;



        /* At the end of the frame, report INT_MAX instead of the height of

         * the frame. This makes the other threads' ff_thread_await_progress()

         * calls cheaper, because they don't have to clip their values. */

        ff_thread_report_progress(&s->current_frame,

                                  y_flipped == s->avctx->height ? INT_MAX

                                                                : y_flipped - 1,

                                  0);

    }



    if (s->avctx->draw_horiz_band == NULL)

        return;



    h = y - s->last_slice_end;

    s->last_slice_end = y;

    y -= h;



    if (!s->flipped_image)

        y = s->avctx->height - y - h;



    cy        = y >> s->chroma_y_shift;

    offset[0] = s->current_frame.f->linesize[0] * y;

    offset[1] = s->current_frame.f->linesize[1] * cy;

    offset[2] = s->current_frame.f->linesize[2] * cy;

    for (i = 3; i < AV_NUM_DATA_POINTERS; i++)

        offset[i] = 0;



    emms_c();

    s->avctx->draw_horiz_band(s->avctx, s->current_frame.f, offset, y, 3, h);

}
