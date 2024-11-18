static void FUNC(hevc_loop_filter_luma)(uint8_t *_pix,

                                        ptrdiff_t _xstride, ptrdiff_t _ystride,

                                        int *_beta, int *_tc,

                                        uint8_t *_no_p, uint8_t *_no_q)

{

    int d, j;

    pixel *pix        = (pixel *)_pix;

    ptrdiff_t xstride = _xstride / sizeof(pixel);

    ptrdiff_t ystride = _ystride / sizeof(pixel);



    for (j = 0; j < 2; j++) {

        const int dp0  = abs(P2  - 2 * P1  + P0);

        const int dq0  = abs(Q2  - 2 * Q1  + Q0);

        const int dp3  = abs(TP2 - 2 * TP1 + TP0);

        const int dq3  = abs(TQ2 - 2 * TQ1 + TQ0);

        const int d0   = dp0 + dq0;

        const int d3   = dp3 + dq3;

        const int beta = _beta[j] << (BIT_DEPTH - 8);

        const int tc   = _tc[j]   << (BIT_DEPTH - 8);

        const int no_p = _no_p[j];

        const int no_q = _no_q[j];



        if (d0 + d3 >= beta) {

            pix += 4 * ystride;

            continue;

        } else {

            const int beta_3 = beta >> 3;

            const int beta_2 = beta >> 2;

            const int tc25   = ((tc * 5 + 1) >> 1);



            if (abs(P3  -  P0) + abs(Q3  -  Q0) < beta_3 && abs(P0  -  Q0) < tc25 &&

                abs(TP3 - TP0) + abs(TQ3 - TQ0) < beta_3 && abs(TP0 - TQ0) < tc25 &&

                                      (d0 << 1) < beta_2 &&      (d3 << 1) < beta_2) {

                // strong filtering

                const int tc2 = tc << 1;

                for (d = 0; d < 4; d++) {

                    const int p3 = P3;

                    const int p2 = P2;

                    const int p1 = P1;

                    const int p0 = P0;

                    const int q0 = Q0;

                    const int q1 = Q1;

                    const int q2 = Q2;

                    const int q3 = Q3;

                    if (!no_p) {

                        P0 = p0 + av_clip(((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) - p0, -tc2, tc2);

                        P1 = p1 + av_clip(((p2 + p1 + p0 + q0 + 2) >> 2) - p1, -tc2, tc2);

                        P2 = p2 + av_clip(((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) - p2, -tc2, tc2);

                    }

                    if (!no_q) {

                        Q0 = q0 + av_clip(((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) - q0, -tc2, tc2);

                        Q1 = q1 + av_clip(((p0 + q0 + q1 + q2 + 2) >> 2) - q1, -tc2, tc2);

                        Q2 = q2 + av_clip(((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) - q2, -tc2, tc2);

                    }

                    pix += ystride;

                }

            } else { // normal filtering

                int nd_p = 1;

                int nd_q = 1;

                const int tc_2 = tc >> 1;

                if (dp0 + dp3 < ((beta + (beta >> 1)) >> 3))

                    nd_p = 2;

                if (dq0 + dq3 < ((beta + (beta >> 1)) >> 3))

                    nd_q = 2;



                for (d = 0; d < 4; d++) {

                    const int p2 = P2;

                    const int p1 = P1;

                    const int p0 = P0;

                    const int q0 = Q0;

                    const int q1 = Q1;

                    const int q2 = Q2;

                    int delta0   = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;

                    if (abs(delta0) < 10 * tc) {

                        delta0 = av_clip(delta0, -tc, tc);

                        if (!no_p)

                            P0 = av_clip_pixel(p0 + delta0);

                        if (!no_q)

                            Q0 = av_clip_pixel(q0 - delta0);

                        if (!no_p && nd_p > 1) {

                            const int deltap1 = av_clip((((p2 + p0 + 1) >> 1) - p1 + delta0) >> 1, -tc_2, tc_2);

                            P1 = av_clip_pixel(p1 + deltap1);

                        }

                        if (!no_q && nd_q > 1) {

                            const int deltaq1 = av_clip((((q2 + q0 + 1) >> 1) - q1 - delta0) >> 1, -tc_2, tc_2);

                            Q1 = av_clip_pixel(q1 + deltaq1);

                        }

                    }

                    pix += ystride;

                }

            }

        }

    }

}
