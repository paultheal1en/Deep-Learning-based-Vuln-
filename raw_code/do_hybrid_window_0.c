static void do_hybrid_window(int order, int n, int non_rec, const float *in,

                             float *out, float *hist, float *out2,

                             const float *window)

{

    int i;

    float buffer1[order + 1];

    float buffer2[order + 1];

    float work[order + n + non_rec];



    /* update history */

    memmove(hist, hist + n, (order + non_rec)*sizeof(*hist));



    for (i=0; i < n; i++)

        hist[order + non_rec + i] = in[n-i-1];



    colmult(work, window, hist, order + n + non_rec);



    convolve(buffer1, work + order    , n      , order);

    convolve(buffer2, work + order + n, non_rec, order);



    for (i=0; i <= order; i++) {

        out2[i] = out2[i] * 0.5625 + buffer1[i];

        out [i] = out2[i]          + buffer2[i];

    }



    /* Multiply by the white noise correcting factor (WNCF) */

    *out *= 257./256.;

}
