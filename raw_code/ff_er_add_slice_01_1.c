void ff_er_add_slice(ERContext *s, int startx, int starty,

                     int endx, int endy, int status)

{

    const int start_i  = av_clip(startx + starty * s->mb_width, 0, s->mb_num - 1);

    const int end_i    = av_clip(endx   + endy   * s->mb_width, 0, s->mb_num);

    const int start_xy = s->mb_index2xy[start_i];

    const int end_xy   = s->mb_index2xy[end_i];

    int mask           = -1;



    if (s->avctx->hwaccel && s->avctx->hwaccel->decode_slice)

        return;



    if (start_i > end_i || start_xy > end_xy) {

        av_log(s->avctx, AV_LOG_ERROR,

               "internal error, slice end before start\n");

        return;

    }



    if (!s->avctx->error_concealment)

        return;



    mask &= ~VP_START;

    if (status & (ER_AC_ERROR | ER_AC_END)) {

        mask           &= ~(ER_AC_ERROR | ER_AC_END);

        s->error_count -= end_i - start_i + 1;

    }

    if (status & (ER_DC_ERROR | ER_DC_END)) {

        mask           &= ~(ER_DC_ERROR | ER_DC_END);

        s->error_count -= end_i - start_i + 1;

    }

    if (status & (ER_MV_ERROR | ER_MV_END)) {

        mask           &= ~(ER_MV_ERROR | ER_MV_END);

        s->error_count -= end_i - start_i + 1;

    }



    if (status & ER_MB_ERROR) {

        s->error_occurred = 1;

        s->error_count    = INT_MAX;

    }



    if (mask == ~0x7F) {

        memset(&s->error_status_table[start_xy], 0,

               (end_xy - start_xy) * sizeof(uint8_t));

    } else {

        int i;

        for (i = start_xy; i < end_xy; i++)

            s->error_status_table[i] &= mask;

    }



    if (end_i == s->mb_num)

        s->error_count = INT_MAX;

    else {

        s->error_status_table[end_xy] &= mask;

        s->error_status_table[end_xy] |= status;

    }



    s->error_status_table[start_xy] |= VP_START;



    if (start_xy > 0 && !(s->avctx->active_thread_type & FF_THREAD_SLICE) &&

        er_supported(s) && s->avctx->skip_top * s->mb_width < start_i) {

        int prev_status = s->error_status_table[s->mb_index2xy[start_i - 1]];



        prev_status &= ~ VP_START;

        if (prev_status != (ER_MV_END | ER_DC_END | ER_AC_END)) {

            s->error_occurred = 1;

            s->error_count = INT_MAX;

        }

    }

}
