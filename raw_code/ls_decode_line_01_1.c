static inline void ls_decode_line(JLSState *state, MJpegDecodeContext *s,

                                  void *last, void *dst, int last2, int w,

                                  int stride, int comp, int bits)

{

    int i, x = 0;

    int Ra, Rb, Rc, Rd;

    int D0, D1, D2;



    while (x < w) {

        int err, pred;



        /* compute gradients */

        Ra = x ? R(dst, x - stride) : R(last, x);

        Rb = R(last, x);

        Rc = x ? R(last, x - stride) : last2;

        Rd = (x >= w - stride) ? R(last, x) : R(last, x + stride);

        D0 = Rd - Rb;

        D1 = Rb - Rc;

        D2 = Rc - Ra;

        /* run mode */

        if ((FFABS(D0) <= state->near) &&

            (FFABS(D1) <= state->near) &&

            (FFABS(D2) <= state->near)) {

            int r;

            int RItype;



            /* decode full runs while available */

            while (get_bits1(&s->gb)) {

                int r;

                r = 1 << ff_log2_run[state->run_index[comp]];

                if (x + r * stride > w)

                    r = (w - x) / stride;

                for (i = 0; i < r; i++) {

                    W(dst, x, Ra);

                    x += stride;

                }

                /* if EOL reached, we stop decoding */

                if (r != 1 << ff_log2_run[state->run_index[comp]])

                    return;

                if (state->run_index[comp] < 31)

                    state->run_index[comp]++;

                if (x + stride > w)

                    return;

            }

            /* decode aborted run */

            r = ff_log2_run[state->run_index[comp]];

            if (r)

                r = get_bits_long(&s->gb, r);

            if (x + r * stride > w) {

                r = (w - x) / stride;

            }

            for (i = 0; i < r; i++) {

                W(dst, x, Ra);

                x += stride;

            }



            if (x >= w) {

                av_log(NULL, AV_LOG_ERROR, "run overflow\n");


                return;

            }



            /* decode run termination value */

            Rb     = R(last, x);

            RItype = (FFABS(Ra - Rb) <= state->near) ? 1 : 0;

            err    = ls_get_code_runterm(&s->gb, state, RItype,

                                         ff_log2_run[state->run_index[comp]]);

            if (state->run_index[comp])

                state->run_index[comp]--;



            if (state->near && RItype) {

                pred = Ra + err;

            } else {

                if (Rb < Ra)

                    pred = Rb - err;

                else

                    pred = Rb + err;

            }

        } else { /* regular mode */

            int context, sign;



            context = ff_jpegls_quantize(state, D0) * 81 +

                      ff_jpegls_quantize(state, D1) *  9 +

                      ff_jpegls_quantize(state, D2);

            pred    = mid_pred(Ra, Ra + Rb - Rc, Rb);



            if (context < 0) {

                context = -context;

                sign    = 1;

            } else {

                sign = 0;

            }



            if (sign) {

                pred = av_clip(pred - state->C[context], 0, state->maxval);

                err  = -ls_get_code_regular(&s->gb, state, context);

            } else {

                pred = av_clip(pred + state->C[context], 0, state->maxval);

                err  = ls_get_code_regular(&s->gb, state, context);

            }



            /* we have to do something more for near-lossless coding */

            pred += err;

        }

        if (state->near) {

            if (pred < -state->near)

                pred += state->range * state->twonear;

            else if (pred > state->maxval + state->near)

                pred -= state->range * state->twonear;

            pred = av_clip(pred, 0, state->maxval);

        }



        pred &= state->maxval;

        W(dst, x, pred);

        x += stride;

    }

}