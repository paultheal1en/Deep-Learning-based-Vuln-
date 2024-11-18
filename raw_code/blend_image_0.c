static void blend_image(AVFilterContext *ctx,

                        AVFilterBufferRef *dst, AVFilterBufferRef *src,

                        int x, int y)

{

    OverlayContext *over = ctx->priv;

    int i, j, k;

    int width   = src->video->w;

    int height  = src->video->h;



    if (over->main_is_packed_rgb) {

        uint8_t *dp = dst->data[0] + x * over->main_pix_step[0] +

                      y * dst->linesize[0];

        uint8_t *sp = src->data[0];

        uint8_t alpha;          ///< the amount of overlay to blend on to main

        const int dr = over->main_rgba_map[R];

        const int dg = over->main_rgba_map[G];

        const int db = over->main_rgba_map[B];

        const int da = over->main_rgba_map[A];

        const int dstep = over->main_pix_step[0];

        const int sr = over->overlay_rgba_map[R];

        const int sg = over->overlay_rgba_map[G];

        const int sb = over->overlay_rgba_map[B];

        const int sa = over->overlay_rgba_map[A];

        const int sstep = over->overlay_pix_step[0];

        const int main_has_alpha = over->main_has_alpha;

        for (i = 0; i < height; i++) {

            uint8_t *d = dp, *s = sp;

            for (j = 0; j < width; j++) {

                alpha = s[sa];



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

                    d[dr] = s[sr];

                    d[dg] = s[sg];

                    d[db] = s[sb];

                    break;

                default:

                    // main_value = main_value * (1 - alpha) + overlay_value * alpha

                    // since alpha is in the range 0-255, the result must divided by 255

                    d[dr] = FAST_DIV255(d[dr] * (255 - alpha) + s[sr] * alpha);

                    d[dg] = FAST_DIV255(d[dg] * (255 - alpha) + s[sg] * alpha);

                    d[db] = FAST_DIV255(d[db] * (255 - alpha) + s[sb] * alpha);

                }

                if (main_has_alpha) {

                    switch (alpha) {

                    case 0:

                        break;

                    case 255:

                        d[da] = s[sa];

                        break;

                    default:

                        // apply alpha compositing: main_alpha += (1-main_alpha) * overlay_alpha

                        d[da] += FAST_DIV255((255 - d[da]) * s[sa]);

                    }

                }

                d += dstep;

                s += sstep;

            }

            dp += dst->linesize[0];

            sp += src->linesize[0];

        }

    } else {

        const int main_has_alpha = over->main_has_alpha;

        if (main_has_alpha) {

            uint8_t *da = dst->data[3] + x * over->main_pix_step[3] +

                          y * dst->linesize[3];

            uint8_t *sa = src->data[3];

            uint8_t alpha;          ///< the amount of overlay to blend on to main

            for (i = 0; i < height; i++) {

                uint8_t *d = da, *s = sa;

                for (j = 0; j < width; j++) {

                    alpha = *s;

                    if (alpha != 0 && alpha != 255) {

                        uint8_t alpha_d = *d;

                        alpha = UNPREMULTIPLY_ALPHA(alpha, alpha_d);

                    }

                    switch (alpha) {

                    case 0:

                        break;

                    case 255:

                        *d = *s;

                        break;

                    default:

                        // apply alpha compositing: main_alpha += (1-main_alpha) * overlay_alpha

                        *d += FAST_DIV255((255 - *d) * *s);

                    }

                    d += 1;

                    s += 1;

                }

                da += dst->linesize[3];

                sa += src->linesize[3];

            }

        }

        for (i = 0; i < 3; i++) {

            int hsub = i ? over->hsub : 0;

            int vsub = i ? over->vsub : 0;

            uint8_t *dp = dst->data[i] + (x >> hsub) +

                (y >> vsub) * dst->linesize[i];

            uint8_t *sp = src->data[i];

            uint8_t *ap = src->data[3];

            int wp = FFALIGN(width, 1<<hsub) >> hsub;

            int hp = FFALIGN(height, 1<<vsub) >> vsub;

            for (j = 0; j < hp; j++) {

                uint8_t *d = dp, *s = sp, *a = ap;

                for (k = 0; k < wp; k++) {

                    // average alpha for color components, improve quality

                    int alpha_v, alpha_h, alpha;

                    if (hsub && vsub && j+1 < hp && k+1 < wp) {

                        alpha = (a[0] + a[src->linesize[3]] +

                                 a[1] + a[src->linesize[3]+1]) >> 2;

                    } else if (hsub || vsub) {

                        alpha_h = hsub && k+1 < wp ?

                            (a[0] + a[1]) >> 1 : a[0];

                        alpha_v = vsub && j+1 < hp ?

                            (a[0] + a[src->linesize[3]]) >> 1 : a[0];

                        alpha = (alpha_v + alpha_h) >> 1;

                    } else

                        alpha = a[0];

                    // if the main channel has an alpha channel, alpha has to be calculated

                    // to create an un-premultiplied (straight) alpha value

                    if (main_has_alpha && alpha != 0 && alpha != 255) {

                        // average alpha for color components, improve quality

                        uint8_t alpha_d;

                        if (hsub && vsub && j+1 < hp && k+1 < wp) {

                            alpha_d = (d[0] + d[src->linesize[3]] +

                                       d[1] + d[src->linesize[3]+1]) >> 2;

                        } else if (hsub || vsub) {

                            alpha_h = hsub && k+1 < wp ?

                                (d[0] + d[1]) >> 1 : d[0];

                            alpha_v = vsub && j+1 < hp ?

                                (d[0] + d[src->linesize[3]]) >> 1 : d[0];

                            alpha_d = (alpha_v + alpha_h) >> 1;

                        } else

                            alpha_d = d[0];

                        alpha = UNPREMULTIPLY_ALPHA(alpha, alpha_d);

                    }

                    *d = FAST_DIV255(*d * (255 - alpha) + *s * alpha);

                    s++;

                    d++;

                    a += 1 << hsub;

                }

                dp += dst->linesize[i];

                sp += src->linesize[i];

                ap += (1 << vsub) * src->linesize[3];

            }

        }

    }

}
