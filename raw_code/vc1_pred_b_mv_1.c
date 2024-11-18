static inline void vc1_pred_b_mv(VC1Context *v, int dmv_x[2], int dmv_y[2],
                                 int direct, int mvtype)
{
    MpegEncContext *s = &v->s;
    int xy, wrap, off = 0;
    int16_t *A, *B, *C;
    int px, py;
    int sum;
    int r_x, r_y;
    const uint8_t *is_intra = v->mb_type[0];
    r_x = v->range_x;
    r_y = v->range_y;
    /* scale MV difference to be quad-pel */
    dmv_x[0] <<= 1 - s->quarter_sample;
    dmv_y[0] <<= 1 - s->quarter_sample;
    dmv_x[1] <<= 1 - s->quarter_sample;
    dmv_y[1] <<= 1 - s->quarter_sample;
    wrap = s->b8_stride;
    xy = s->block_index[0];
    if (s->mb_intra) {
        s->current_picture.motion_val[0][xy + v->blocks_off][0] =
        s->current_picture.motion_val[0][xy + v->blocks_off][1] =
        s->current_picture.motion_val[1][xy + v->blocks_off][0] =
        s->current_picture.motion_val[1][xy + v->blocks_off][1] = 0;
        return;
    }
    if (!v->field_mode) {
        s->mv[0][0][0] = scale_mv(s->next_picture.motion_val[1][xy][0], v->bfraction, 0, s->quarter_sample);
        s->mv[0][0][1] = scale_mv(s->next_picture.motion_val[1][xy][1], v->bfraction, 0, s->quarter_sample);
        s->mv[1][0][0] = scale_mv(s->next_picture.motion_val[1][xy][0], v->bfraction, 1, s->quarter_sample);
        s->mv[1][0][1] = scale_mv(s->next_picture.motion_val[1][xy][1], v->bfraction, 1, s->quarter_sample);
        /* Pullback predicted motion vectors as specified in 8.4.5.4 */
        s->mv[0][0][0] = av_clip(s->mv[0][0][0], -60 - (s->mb_x << 6), (s->mb_width  << 6) - 4 - (s->mb_x << 6));
        s->mv[0][0][1] = av_clip(s->mv[0][0][1], -60 - (s->mb_y << 6), (s->mb_height << 6) - 4 - (s->mb_y << 6));
        s->mv[1][0][0] = av_clip(s->mv[1][0][0], -60 - (s->mb_x << 6), (s->mb_width  << 6) - 4 - (s->mb_x << 6));
        s->mv[1][0][1] = av_clip(s->mv[1][0][1], -60 - (s->mb_y << 6), (s->mb_height << 6) - 4 - (s->mb_y << 6));
    }
    if (direct) {
        s->current_picture.motion_val[0][xy + v->blocks_off][0] = s->mv[0][0][0];
        s->current_picture.motion_val[0][xy + v->blocks_off][1] = s->mv[0][0][1];
        s->current_picture.motion_val[1][xy + v->blocks_off][0] = s->mv[1][0][0];
        s->current_picture.motion_val[1][xy + v->blocks_off][1] = s->mv[1][0][1];
        return;
    }
    if ((mvtype == BMV_TYPE_FORWARD) || (mvtype == BMV_TYPE_INTERPOLATED)) {
        C   = s->current_picture.motion_val[0][xy - 2];
        A   = s->current_picture.motion_val[0][xy - wrap * 2];
        off = (s->mb_x == (s->mb_width - 1)) ? -2 : 2;
        B   = s->current_picture.motion_val[0][xy - wrap * 2 + off];
        if (!s->mb_x) C[0] = C[1] = 0;
        if (!s->first_slice_line) { // predictor A is not out of bounds
            if (s->mb_width == 1) {
                px = A[0];
                py = A[1];
            } else {
                px = mid_pred(A[0], B[0], C[0]);
                py = mid_pred(A[1], B[1], C[1]);
            }
        } else if (s->mb_x) { // predictor C is not out of bounds
            px = C[0];
            py = C[1];
        } else {
            px = py = 0;
        }
        /* Pullback MV as specified in 8.3.5.3.4 */
        {
            int qx, qy, X, Y;
            if (v->profile < PROFILE_ADVANCED) {
                qx = (s->mb_x << 5);
                qy = (s->mb_y << 5);
                X  = (s->mb_width  << 5) - 4;
                Y  = (s->mb_height << 5) - 4;
                if (qx + px < -28) px = -28 - qx;
                if (qy + py < -28) py = -28 - qy;
                if (qx + px > X) px = X - qx;
                if (qy + py > Y) py = Y - qy;
            } else {
                qx = (s->mb_x << 6);
                qy = (s->mb_y << 6);
                X  = (s->mb_width  << 6) - 4;
                Y  = (s->mb_height << 6) - 4;
                if (qx + px < -60) px = -60 - qx;
                if (qy + py < -60) py = -60 - qy;
                if (qx + px > X) px = X - qx;
                if (qy + py > Y) py = Y - qy;
            }
        }
        /* Calculate hybrid prediction as specified in 8.3.5.3.5 */
        if (0 && !s->first_slice_line && s->mb_x) {
            if (is_intra[xy - wrap])
                sum = FFABS(px) + FFABS(py);
            else
                sum = FFABS(px - A[0]) + FFABS(py - A[1]);
            if (sum > 32) {
                if (get_bits1(&s->gb)) {
                    px = A[0];
                    py = A[1];
                } else {
                    px = C[0];
                    py = C[1];
                }
            } else {
                if (is_intra[xy - 2])
                    sum = FFABS(px) + FFABS(py);
                else
                    sum = FFABS(px - C[0]) + FFABS(py - C[1]);
                if (sum > 32) {
                    if (get_bits1(&s->gb)) {
                        px = A[0];
                        py = A[1];
                    } else {
                        px = C[0];
                        py = C[1];
                    }
                }
            }
        }
        /* store MV using signed modulus of MV range defined in 4.11 */
        s->mv[0][0][0] = ((px + dmv_x[0] + r_x) & ((r_x << 1) - 1)) - r_x;
        s->mv[0][0][1] = ((py + dmv_y[0] + r_y) & ((r_y << 1) - 1)) - r_y;
    }
    if ((mvtype == BMV_TYPE_BACKWARD) || (mvtype == BMV_TYPE_INTERPOLATED)) {
        C   = s->current_picture.motion_val[1][xy - 2];
        A   = s->current_picture.motion_val[1][xy - wrap * 2];
        off = (s->mb_x == (s->mb_width - 1)) ? -2 : 2;
        B   = s->current_picture.motion_val[1][xy - wrap * 2 + off];
        if (!s->mb_x)
            C[0] = C[1] = 0;
        if (!s->first_slice_line) { // predictor A is not out of bounds
            if (s->mb_width == 1) {
                px = A[0];
                py = A[1];
            } else {
                px = mid_pred(A[0], B[0], C[0]);
                py = mid_pred(A[1], B[1], C[1]);
            }
        } else if (s->mb_x) { // predictor C is not out of bounds
            px = C[0];
            py = C[1];
        } else {
            px = py = 0;
        }
        /* Pullback MV as specified in 8.3.5.3.4 */
        {
            int qx, qy, X, Y;
            if (v->profile < PROFILE_ADVANCED) {
                qx = (s->mb_x << 5);
                qy = (s->mb_y << 5);
                X  = (s->mb_width  << 5) - 4;
                Y  = (s->mb_height << 5) - 4;
                if (qx + px < -28) px = -28 - qx;
                if (qy + py < -28) py = -28 - qy;
                if (qx + px > X) px = X - qx;
                if (qy + py > Y) py = Y - qy;
            } else {
                qx = (s->mb_x << 6);
                qy = (s->mb_y << 6);
                X  = (s->mb_width  << 6) - 4;
                Y  = (s->mb_height << 6) - 4;
                if (qx + px < -60) px = -60 - qx;
                if (qy + py < -60) py = -60 - qy;
                if (qx + px > X) px = X - qx;
                if (qy + py > Y) py = Y - qy;
            }
        }
        /* Calculate hybrid prediction as specified in 8.3.5.3.5 */
        if (0 && !s->first_slice_line && s->mb_x) {
            if (is_intra[xy - wrap])
                sum = FFABS(px) + FFABS(py);
            else
                sum = FFABS(px - A[0]) + FFABS(py - A[1]);
            if (sum > 32) {
                if (get_bits1(&s->gb)) {
                    px = A[0];
                    py = A[1];
                } else {
                    px = C[0];
                    py = C[1];
                }
            } else {
                if (is_intra[xy - 2])
                    sum = FFABS(px) + FFABS(py);
                else
                    sum = FFABS(px - C[0]) + FFABS(py - C[1]);
                if (sum > 32) {
                    if (get_bits1(&s->gb)) {
                        px = A[0];
                        py = A[1];
                    } else {
                        px = C[0];
                        py = C[1];
                    }
                }
            }
        }
        /* store MV using signed modulus of MV range defined in 4.11 */
        s->mv[1][0][0] = ((px + dmv_x[1] + r_x) & ((r_x << 1) - 1)) - r_x;
        s->mv[1][0][1] = ((py + dmv_y[1] + r_y) & ((r_y << 1) - 1)) - r_y;
    }
    s->current_picture.motion_val[0][xy][0] = s->mv[0][0][0];
    s->current_picture.motion_val[0][xy][1] = s->mv[0][0][1];
    s->current_picture.motion_val[1][xy][0] = s->mv[1][0][0];
    s->current_picture.motion_val[1][xy][1] = s->mv[1][0][1];
}