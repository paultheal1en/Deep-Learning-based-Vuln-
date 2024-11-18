static void rgb24_to_yuvj420p(AVPicture *dst, AVPicture *src,

                              int width, int height)

{

    int wrap, wrap3, width2;

    int r, g, b, r1, g1, b1, w;

    uint8_t *lum, *cb, *cr;

    const uint8_t *p;



    lum = dst->data[0];

    cb = dst->data[1];

    cr = dst->data[2];



    width2 = (width + 1) >> 1;

    wrap = dst->linesize[0];

    wrap3 = src->linesize[0];

    p = src->data[0];

    for(;height>=2;height -= 2) {

        for(w = width; w >= 2; w -= 2) {

            RGB_IN(r, g, b, p);

            r1 = r;

            g1 = g;

            b1 = b;

            lum[0] = RGB_TO_Y(r, g, b);



            RGB_IN(r, g, b, p + BPP);

            r1 += r;

            g1 += g;

            b1 += b;

            lum[1] = RGB_TO_Y(r, g, b);

            p += wrap3;

            lum += wrap;



            RGB_IN(r, g, b, p);

            r1 += r;

            g1 += g;

            b1 += b;

            lum[0] = RGB_TO_Y(r, g, b);



            RGB_IN(r, g, b, p + BPP);

            r1 += r;

            g1 += g;

            b1 += b;

            lum[1] = RGB_TO_Y(r, g, b);



            cb[0] = RGB_TO_U(r1, g1, b1, 2);

            cr[0] = RGB_TO_V(r1, g1, b1, 2);



            cb++;

            cr++;

            p += -wrap3 + 2 * BPP;

            lum += -wrap + 2;

        }

        if (w) {

            RGB_IN(r, g, b, p);

            r1 = r;

            g1 = g;

            b1 = b;

            lum[0] = RGB_TO_Y(r, g, b);

            p += wrap3;

            lum += wrap;

            RGB_IN(r, g, b, p);

            r1 += r;

            g1 += g;

            b1 += b;

            lum[0] = RGB_TO_Y(r, g, b);

            cb[0] = RGB_TO_U(r1, g1, b1, 1);

            cr[0] = RGB_TO_V(r1, g1, b1, 1);

            cb++;

            cr++;

            p += -wrap3 + BPP;

            lum += -wrap + 1;

        }

        p += wrap3 + (wrap3 - width * BPP);

        lum += wrap + (wrap - width);

        cb += dst->linesize[1] - width2;

        cr += dst->linesize[2] - width2;

    }

    /* handle odd height */

    if (height) {

        for(w = width; w >= 2; w -= 2) {

            RGB_IN(r, g, b, p);

            r1 = r;

            g1 = g;

            b1 = b;

            lum[0] = RGB_TO_Y(r, g, b);



            RGB_IN(r, g, b, p + BPP);

            r1 += r;

            g1 += g;

            b1 += b;

            lum[1] = RGB_TO_Y(r, g, b);

            cb[0] = RGB_TO_U(r1, g1, b1, 1);

            cr[0] = RGB_TO_V(r1, g1, b1, 1);

            cb++;

            cr++;

            p += 2 * BPP;

           lum += 2;

        }

        if (w) {

            RGB_IN(r, g, b, p);

            lum[0] = RGB_TO_Y(r, g, b);

            cb[0] = RGB_TO_U(r, g, b, 0);

            cr[0] = RGB_TO_V(r, g, b, 0);

        }

    }

}
