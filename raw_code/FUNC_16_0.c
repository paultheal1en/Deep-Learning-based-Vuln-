static void FUNC(hevc_loop_filter_chroma)(uint8_t *_pix, ptrdiff_t _xstride,

                                          ptrdiff_t _ystride, int *_tc,

                                          uint8_t *_no_p, uint8_t *_no_q)

{

    int d, j, no_p, no_q;

    pixel *pix        = (pixel *)_pix;

    ptrdiff_t xstride = _xstride / sizeof(pixel);

    ptrdiff_t ystride = _ystride / sizeof(pixel);



    for (j = 0; j < 2; j++) {

        const int tc = _tc[j] << (BIT_DEPTH - 8);

        if (tc <= 0) {

            pix += 4 * ystride;

            continue;

        }

        no_p = _no_p[j];

        no_q = _no_q[j];



        for (d = 0; d < 4; d++) {

            int delta0;

            const int p1 = P1;

            const int p0 = P0;

            const int q0 = Q0;

            const int q1 = Q1;

            delta0 = av_clip((((q0 - p0) << 2) + p1 - q1 + 4) >> 3, -tc, tc);

            if (!no_p)

                P0 = av_clip_pixel(p0 + delta0);

            if (!no_q)

                Q0 = av_clip_pixel(q0 - delta0);

            pix += ystride;

        }

    }

}
