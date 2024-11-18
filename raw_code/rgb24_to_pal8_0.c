static void rgb24_to_pal8(AVPicture *dst, AVPicture *src,

                          int width, int height)

{

    const unsigned char *p;

    unsigned char *q;

    int r, g, b, dst_wrap, src_wrap;

    int x, y, i;

    static const uint8_t pal_value[6] = { 0x00, 0x33, 0x66, 0x99, 0xcc, 0xff };

    uint32_t *pal;



    p = src->data[0];

    src_wrap = src->linesize[0] - 3 * width;



    q = dst->data[0];

    dst_wrap = dst->linesize[0] - width;



    for(y=0;y<height;y++) {

        for(x=0;x<width;x++) {

            r = p[0];

            g = p[1];

            b = p[2];



            q[0] = gif_clut_index(r, g, b);

            q++;

            p += 3;

        }

        p += src_wrap;

        q += dst_wrap;

    }



    /* build palette */

    pal = (uint32_t *)dst->data[1];

    i = 0;

    for(r = 0; r < 6; r++) {

        for(g = 0; g < 6; g++) {

            for(b = 0; b < 6; b++) {

                pal[i++] = (0xff << 24) | (pal_value[r] << 16) | 

                    (pal_value[g] << 8) | pal_value[b];

            }

        }

    }

    while (i < 256)

        pal[i++] = 0;

}
