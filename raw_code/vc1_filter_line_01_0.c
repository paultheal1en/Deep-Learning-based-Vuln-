static int vc1_filter_line(uint8_t* src, int stride, int pq){

    int a0, a1, a2, a3, d, clip, filt3 = 0;

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    a0     = (2*(src[-2*stride] - src[ 1*stride]) - 5*(src[-1*stride] - src[ 0*stride]) + 4) >> 3;

    if(FFABS(a0) < pq){

        a1 = (2*(src[-4*stride] - src[-1*stride]) - 5*(src[-3*stride] - src[-2*stride]) + 4) >> 3;

        a2 = (2*(src[ 0*stride] - src[ 3*stride]) - 5*(src[ 1*stride] - src[ 2*stride]) + 4) >> 3;

        a3 = FFMIN(FFABS(a1), FFABS(a2));

        if(a3 < FFABS(a0)){

            d = 5 * ((a0 >=0 ? a3 : -a3) - a0) / 8;

            clip = (src[-1*stride] - src[ 0*stride])/2;

            if(clip){

                filt3 = 1;

                if(clip > 0)

                    d = av_clip(d, 0, clip);

                else

                    d = av_clip(d, clip, 0);

                src[-1*stride] = cm[src[-1*stride] - d];

                src[ 0*stride] = cm[src[ 0*stride] + d];

            }

        }

    }

    return filt3;

}
