static void rgba32_to_rgb24(AVPicture *dst, AVPicture *src,

                            int width, int height)

{

    const uint8_t *s;

    uint8_t *d;

    int src_wrap, dst_wrap, j, y;

    unsigned int v;



    s = src->data[0];

    src_wrap = src->linesize[0] - width * 4;



    d = dst->data[0];

    dst_wrap = dst->linesize[0] - width * 3;



    for(y=0;y<height;y++) {

        for(j = 0;j < width; j++) {

            v = *(uint32_t *)s;

            s += 4;

            d[0] = v >> 16;

            d[1] = v >> 8;

            d[2] = v;

            d += 3;

        }

        s += src_wrap;

        d += dst_wrap;

    }

}
