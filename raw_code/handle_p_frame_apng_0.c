static int handle_p_frame_apng(AVCodecContext *avctx, PNGDecContext *s,

                               AVFrame *p)

{

    int i, j;

    uint8_t *pd      = p->data[0];

    uint8_t *pd_last = s->last_picture.f->data[0];

    uint8_t *pd_last_region = s->dispose_op == APNG_DISPOSE_OP_PREVIOUS ?

                                s->previous_picture.f->data[0] : s->last_picture.f->data[0];

    int ls = FFMIN(av_image_get_linesize(p->format, s->width, 0), s->width * s->bpp);



    if (s->blend_op == APNG_BLEND_OP_OVER &&

        avctx->pix_fmt != AV_PIX_FMT_RGBA && avctx->pix_fmt != AV_PIX_FMT_ARGB) {

        avpriv_request_sample(avctx, "Blending with pixel format %s",

                              av_get_pix_fmt_name(avctx->pix_fmt));

        return AVERROR_PATCHWELCOME;

    }



    ff_thread_await_progress(&s->last_picture, INT_MAX, 0);

    if (s->dispose_op == APNG_DISPOSE_OP_PREVIOUS)

        ff_thread_await_progress(&s->previous_picture, INT_MAX, 0);



    for (j = 0; j < s->y_offset; j++) {

        for (i = 0; i < ls; i++)

            pd[i] = pd_last[i];

        pd      += s->image_linesize;

        pd_last += s->image_linesize;

    }



    if (s->dispose_op != APNG_DISPOSE_OP_BACKGROUND && s->blend_op == APNG_BLEND_OP_OVER) {

        uint8_t ri, gi, bi, ai;



        pd_last_region += s->y_offset * s->image_linesize;

        if (avctx->pix_fmt == AV_PIX_FMT_RGBA) {

            ri = 0;

            gi = 1;

            bi = 2;

            ai = 3;

        } else {

            ri = 3;

            gi = 2;

            bi = 1;

            ai = 0;

        }



        for (j = s->y_offset; j < s->y_offset + s->cur_h; j++) {

            for (i = 0; i < s->x_offset * s->bpp; i++)

                pd[i] = pd_last[i];

            for (; i < (s->x_offset + s->cur_w) * s->bpp; i += s->bpp) {

                uint8_t alpha = pd[i+ai];



                /* output = alpha * foreground + (1-alpha) * background */

                switch (alpha) {

                case 0:

                    pd[i+ri] = pd_last_region[i+ri];

                    pd[i+gi] = pd_last_region[i+gi];

                    pd[i+bi] = pd_last_region[i+bi];

                    pd[i+ai] = 0xff;

                    break;

                case 255:

                    break;

                default:

                    pd[i+ri] = FAST_DIV255(alpha * pd[i+ri] + (255 - alpha) * pd_last_region[i+ri]);

                    pd[i+gi] = FAST_DIV255(alpha * pd[i+gi] + (255 - alpha) * pd_last_region[i+gi]);

                    pd[i+bi] = FAST_DIV255(alpha * pd[i+bi] + (255 - alpha) * pd_last_region[i+bi]);

                    pd[i+ai] = 0xff;

                    break;

                }

            }

            for (; i < ls; i++)

                pd[i] = pd_last[i];

            pd      += s->image_linesize;

            pd_last += s->image_linesize;

            pd_last_region += s->image_linesize;

        }

    } else {

        for (j = s->y_offset; j < s->y_offset + s->cur_h; j++) {

            for (i = 0; i < s->x_offset * s->bpp; i++)

                pd[i] = pd_last[i];

            for (i = (s->x_offset + s->cur_w) * s->bpp; i < ls; i++)

                pd[i] = pd_last[i];

            pd      += s->image_linesize;

            pd_last += s->image_linesize;

        }

    }



    for (j = s->y_offset + s->cur_h; j < s->height; j++) {

        for (i = 0; i < ls; i++)

            pd[i] = pd_last[i];

        pd      += s->image_linesize;

        pd_last += s->image_linesize;

    }



    return 0;

}
