static inline int vc1_pred_dc(MpegEncContext *s, int overlap, int pq, int n,

                              int a_avail, int c_avail,

                              int16_t **dc_val_ptr, int *dir_ptr)

{

    int a, b, c, wrap, pred;

    int16_t *dc_val;

    int mb_pos = s->mb_x + s->mb_y * s->mb_stride;

    int q1, q2 = 0;



    wrap = s->block_wrap[n];

    dc_val = s->dc_val[0] + s->block_index[n];



    /* B A

     * C X

     */

    c = dc_val[ - 1];

    b = dc_val[ - 1 - wrap];

    a = dc_val[ - wrap];

    /* scale predictors if needed */

    q1 = s->current_picture.f.qscale_table[mb_pos];

    if (c_avail && (n != 1 && n != 3)) {

        q2 = s->current_picture.f.qscale_table[mb_pos - 1];

        if (q2 && q2 != q1)

            c = (c * s->y_dc_scale_table[q2] * ff_vc1_dqscale[s->y_dc_scale_table[q1] - 1] + 0x20000) >> 18;

    }

    if (a_avail && (n != 2 && n != 3)) {

        q2 = s->current_picture.f.qscale_table[mb_pos - s->mb_stride];

        if (q2 && q2 != q1)

            a = (a * s->y_dc_scale_table[q2] * ff_vc1_dqscale[s->y_dc_scale_table[q1] - 1] + 0x20000) >> 18;

    }

    if (a_avail && c_avail && (n != 3)) {

        int off = mb_pos;

        if (n != 1)

            off--;

        if (n != 2)

            off -= s->mb_stride;

        q2 = s->current_picture.f.qscale_table[off];

        if (q2 && q2 != q1)

            b = (b * s->y_dc_scale_table[q2] * ff_vc1_dqscale[s->y_dc_scale_table[q1] - 1] + 0x20000) >> 18;

    }



    if (a_avail && c_avail) {

        if (abs(a - b) <= abs(b - c)) {

            pred     = c;

            *dir_ptr = 1; // left

        } else {

            pred     = a;

            *dir_ptr = 0; // top

        }

    } else if (a_avail) {

        pred     = a;

        *dir_ptr = 0; // top

    } else if (c_avail) {

        pred     = c;

        *dir_ptr = 1; // left

    } else {

        pred     = 0;

        *dir_ptr = 1; // left

    }



    /* update predictor */

    *dc_val_ptr = &dc_val[0];

    return pred;

}
