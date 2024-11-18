static inline void rv40_weak_loop_filter(uint8_t *src, const int step,

                                         const int filter_p1, const int filter_q1,

                                         const int alpha, const int beta,

                                         const int lim_p0q0,

                                         const int lim_q1, const int lim_p1,

                                         const int diff_p1p0, const int diff_q1q0,

                                         const int diff_p1p2, const int diff_q1q2)

{

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    int t, u, diff;



    t = src[0*step] - src[-1*step];

    if(!t)

        return;

    u = (alpha * FFABS(t)) >> 7;

    if(u > 3 - (filter_p1 && filter_q1))

        return;



    t <<= 2;

    if(filter_p1 && filter_q1)

        t += src[-2*step] - src[1*step];

    diff = CLIP_SYMM((t + 4) >> 3, lim_p0q0);

    src[-1*step] = cm[src[-1*step] + diff];

    src[ 0*step] = cm[src[ 0*step] - diff];

    if(FFABS(diff_p1p2) <= beta && filter_p1){

        t = (diff_p1p0 + diff_p1p2 - diff) >> 1;

        src[-2*step] = cm[src[-2*step] - CLIP_SYMM(t, lim_p1)];

    }

    if(FFABS(diff_q1q2) <= beta && filter_q1){

        t = (diff_q1q0 + diff_q1q2 + diff) >> 1;

        src[ 1*step] = cm[src[ 1*step] - CLIP_SYMM(t, lim_q1)];

    }

}
