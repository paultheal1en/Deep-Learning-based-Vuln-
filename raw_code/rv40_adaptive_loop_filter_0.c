static av_always_inline void rv40_adaptive_loop_filter(uint8_t *src, const int step,

                                             const int stride, const int dmode,

                                             const int lim_q1, const int lim_p1,

                                             const int alpha,

                                             const int beta, const int beta2,

                                             const int chroma, const int edge)

{

    int diff_p1p0[4], diff_q1q0[4], diff_p1p2[4], diff_q1q2[4];

    int sum_p1p0 = 0, sum_q1q0 = 0, sum_p1p2 = 0, sum_q1q2 = 0;

    uint8_t *ptr;

    int flag_strong0 = 1, flag_strong1 = 1;

    int filter_p1, filter_q1;

    int i;

    int lims;



    for(i = 0, ptr = src; i < 4; i++, ptr += stride){

        diff_p1p0[i] = ptr[-2*step] - ptr[-1*step];

        diff_q1q0[i] = ptr[ 1*step] - ptr[ 0*step];

        sum_p1p0 += diff_p1p0[i];

        sum_q1q0 += diff_q1q0[i];

    }

    filter_p1 = FFABS(sum_p1p0) < (beta<<2);

    filter_q1 = FFABS(sum_q1q0) < (beta<<2);

    if(!filter_p1 && !filter_q1)

        return;



    for(i = 0, ptr = src; i < 4; i++, ptr += stride){

        diff_p1p2[i] = ptr[-2*step] - ptr[-3*step];

        diff_q1q2[i] = ptr[ 1*step] - ptr[ 2*step];

        sum_p1p2 += diff_p1p2[i];

        sum_q1q2 += diff_q1q2[i];

    }



    if(edge){

        flag_strong0 = filter_p1 && (FFABS(sum_p1p2) < beta2);

        flag_strong1 = filter_q1 && (FFABS(sum_q1q2) < beta2);

    }else{

        flag_strong0 = flag_strong1 = 0;

    }



    lims = filter_p1 + filter_q1 + ((lim_q1 + lim_p1) >> 1) + 1;

    if(flag_strong0 && flag_strong1){ /* strong filtering */

        for(i = 0; i < 4; i++, src += stride){

            int sflag, p0, q0, p1, q1;

            int t = src[0*step] - src[-1*step];



            if(!t) continue;

            sflag = (alpha * FFABS(t)) >> 7;

            if(sflag > 1) continue;



            p0 = (25*src[-3*step] + 26*src[-2*step]

                + 26*src[-1*step]

                + 26*src[ 0*step] + 25*src[ 1*step] + rv40_dither_l[dmode + i]) >> 7;

            q0 = (25*src[-2*step] + 26*src[-1*step]

                + 26*src[ 0*step]

                + 26*src[ 1*step] + 25*src[ 2*step] + rv40_dither_r[dmode + i]) >> 7;

            if(sflag){

                p0 = av_clip(p0, src[-1*step] - lims, src[-1*step] + lims);

                q0 = av_clip(q0, src[ 0*step] - lims, src[ 0*step] + lims);

            }

            p1 = (25*src[-4*step] + 26*src[-3*step]

                + 26*src[-2*step]

                + 26*p0           + 25*src[ 0*step] + rv40_dither_l[dmode + i]) >> 7;

            q1 = (25*src[-1*step] + 26*q0

                + 26*src[ 1*step]

                + 26*src[ 2*step] + 25*src[ 3*step] + rv40_dither_r[dmode + i]) >> 7;

            if(sflag){

                p1 = av_clip(p1, src[-2*step] - lims, src[-2*step] + lims);

                q1 = av_clip(q1, src[ 1*step] - lims, src[ 1*step] + lims);

            }

            src[-2*step] = p1;

            src[-1*step] = p0;

            src[ 0*step] = q0;

            src[ 1*step] = q1;

            if(!chroma){

                src[-3*step] = (25*src[-1*step] + 26*src[-2*step] + 51*src[-3*step] + 26*src[-4*step] + 64) >> 7;

                src[ 2*step] = (25*src[ 0*step] + 26*src[ 1*step] + 51*src[ 2*step] + 26*src[ 3*step] + 64) >> 7;

            }

        }

    }else if(filter_p1 && filter_q1){

        for(i = 0; i < 4; i++, src += stride)

            rv40_weak_loop_filter(src, step, 1, 1, alpha, beta, lims, lim_q1, lim_p1,

                                  diff_p1p0[i], diff_q1q0[i], diff_p1p2[i], diff_q1q2[i]);

    }else{

        for(i = 0; i < 4; i++, src += stride)

            rv40_weak_loop_filter(src, step, filter_p1, filter_q1,

                                  alpha, beta, lims>>1, lim_q1>>1, lim_p1>>1,

                                  diff_p1p0[i], diff_q1q0[i], diff_p1p2[i], diff_q1q2[i]);

    }

}
