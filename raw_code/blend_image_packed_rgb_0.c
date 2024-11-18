static av_always_inline void blend_image_packed_rgb(AVFilterContext *ctx,

                                   AVFrame *dst, const AVFrame *src,

                                   int main_has_alpha, int x, int y,

                                   int is_straight)

{

    OverlayContext *s = ctx->priv;

    int i, imax, j, jmax;

    const int src_w = src->width;

    const int src_h = src->height;

    const int dst_w = dst->width;

    const int dst_h = dst->height;

    uint8_t alpha;          ///< the amount of overlay to blend on to main

    const int dr = s->main_rgba_map[R];

    const int dg = s->main_rgba_map[G];

    const int db = s->main_rgba_map[B];

    const int da = s->main_rgba_map[A];

    const int dstep = s->main_pix_step[0];

    const int sr = s->overlay_rgba_map[R];

    const int sg = s->overlay_rgba_map[G];

    const int sb = s->overlay_rgba_map[B];

    const int sa = s->overlay_rgba_map[A];

    const int sstep = s->overlay_pix_step[0];

    uint8_t *S, *sp, *d, *dp;



    i = FFMAX(-y, 0);

    sp = src->data[0] + i     * src->linesize[0];

    dp = dst->data[0] + (y+i) * dst->linesize[0];



    for (imax = FFMIN(-y + dst_h, src_h); i < imax; i++) {

        j = FFMAX(-x, 0);

        S = sp + j     * sstep;

        d = dp + (x+j) * dstep;



        for (jmax = FFMIN(-x + dst_w, src_w); j < jmax; j++) {

            alpha = S[sa];



            // if the main channel has an alpha channel, alpha has to be calculated

            // to create an un-premultiplied (straight) alpha value

            if (main_has_alpha && alpha != 0 && alpha != 255) {

                uint8_t alpha_d = d[da];

                alpha = UNPREMULTIPLY_ALPHA(alpha, alpha_d);

            }



            switch (alpha) {

            case 0:

                break;

            case 255:

                d[dr] = S[sr];

                d[dg] = S[sg];

                d[db] = S[sb];

                break;

            default:

                // main_value = main_value * (1 - alpha) + overlay_value * alpha

                // since alpha is in the range 0-255, the result must divided by 255

                d[dr] = is_straight ? FAST_DIV255(d[dr] * (255 - alpha) + S[sr] * alpha) : FAST_DIV255(d[dr] * (255 - alpha) + S[sr]);

                d[dg] = is_straight ? FAST_DIV255(d[dg] * (255 - alpha) + S[sg] * alpha) : FAST_DIV255(d[dr] * (255 - alpha) + S[sr]);

                d[db] = is_straight ? FAST_DIV255(d[db] * (255 - alpha) + S[sb] * alpha) : FAST_DIV255(d[dr] * (255 - alpha) + S[sr]);

            }

            if (main_has_alpha) {

                switch (alpha) {

                case 0:

                    break;

                case 255:

                    d[da] = S[sa];

                    break;

                default:

                    // apply alpha compositing: main_alpha += (1-main_alpha) * overlay_alpha

                    d[da] += FAST_DIV255((255 - d[da]) * S[sa]);

                }

            }

            d += dstep;

            S += sstep;

        }

        dp += dst->linesize[0];

        sp += src->linesize[0];

    }

}
