static void draw_rectangle(unsigned val, uint8_t *dst, int dst_linesize, unsigned segment_width,

                           unsigned x, unsigned y, unsigned w, unsigned h)

{

    int i;

    int step = 3;



    dst += segment_width * (step * x + y * dst_linesize);

    w *= segment_width * step;

    h *= segment_width;

    for (i = 0; i < h; i++) {

        memset(dst, val, w);

        dst += dst_linesize;

    }

}
