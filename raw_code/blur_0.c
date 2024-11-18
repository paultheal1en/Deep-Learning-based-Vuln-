static void blur(uint8_t       *dst, const int dst_linesize,

                 const uint8_t *src, const int src_linesize,

                 const int w, const int h, FilterParam *fp)

{

    int x, y;

    FilterParam f = *fp;

    const int radius = f.dist_width/2;



    const uint8_t * const src2[NB_PLANES] = { src };

    int          src2_linesize[NB_PLANES] = { src_linesize };

    uint8_t     *dst2[NB_PLANES] = { f.pre_filter_buf };

    int dst2_linesize[NB_PLANES] = { f.pre_filter_linesize };



    sws_scale(f.pre_filter_context, src2, src2_linesize, 0, h, dst2, dst2_linesize);



#define UPDATE_FACTOR do {                                              \

        int factor;                                                     \

        factor = f.color_diff_coeff[COLOR_DIFF_COEFF_SIZE/2 + pre_val - \

                 f.pre_filter_buf[ix + iy*f.pre_filter_linesize]] * f.dist_coeff[dx + dy*f.dist_linesize]; \

        sum += src[ix + iy*src_linesize] * factor;                      \

        div += factor;                                                  \

    } while (0)



    for (y = 0; y < h; y++) {

        for (x = 0; x < w; x++) {

            int sum = 0;

            int div = 0;

            int dy;

            const int pre_val = f.pre_filter_buf[x + y*f.pre_filter_linesize];

            if (x >= radius && x < w - radius) {

                for (dy = 0; dy < radius*2 + 1; dy++) {

                    int dx;

                    int iy = y+dy - radius;

                    if      (iy < 0)  iy = -iy;

                    else if (iy >= h) iy = h+h-iy-1;



                    for (dx = 0; dx < radius*2 + 1; dx++) {

                        const int ix = x+dx - radius;

                        UPDATE_FACTOR;

                    }

                }

            } else {

                for (dy = 0; dy < radius*2+1; dy++) {

                    int dx;

                    int iy = y+dy - radius;

                    if      (iy <  0) iy = -iy;

                    else if (iy >= h) iy = h+h-iy-1;



                    for (dx = 0; dx < radius*2 + 1; dx++) {

                        int ix = x+dx - radius;

                        if      (ix < 0)  ix = -ix;

                        else if (ix >= w) ix = w+w-ix-1;

                        UPDATE_FACTOR;

                    }

                }

            }

            dst[x + y*dst_linesize] = (sum + div/2) / div;

        }

    }

}
