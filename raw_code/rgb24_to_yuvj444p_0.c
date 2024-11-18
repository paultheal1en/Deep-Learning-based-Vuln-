static void rgb24_to_yuvj444p(AVPicture *dst, AVPicture *src,

                              int width, int height)

{

    int src_wrap, x, y;

    int r, g, b;

    uint8_t *lum, *cb, *cr;

    const uint8_t *p;



    lum = dst->data[0];

    cb = dst->data[1];

    cr = dst->data[2];



    src_wrap = src->linesize[0] - width * BPP;

    p = src->data[0];

    for(y=0;y<height;y++) {

        for(x=0;x<width;x++) {

            RGB_IN(r, g, b, p);

            lum[0] = RGB_TO_Y(r, g, b);

            cb[0] = RGB_TO_U(r, g, b, 0);

            cr[0] = RGB_TO_V(r, g, b, 0);

            cb++;

            cr++;

            lum++;

        }

        p += src_wrap;

        lum += dst->linesize[0] - width;

        cb += dst->linesize[1] - width;

        cr += dst->linesize[2] - width;

    }

}
