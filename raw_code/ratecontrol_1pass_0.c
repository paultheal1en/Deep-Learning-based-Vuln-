static int ratecontrol_1pass(SnowContext *s, AVFrame *pict)

{

    /* estimate the frame's complexity as a sum of weighted dwt coefs.

     * FIXME we know exact mv bits at this point,

     * but ratecontrol isn't set up to include them. */

    uint32_t coef_sum= 0;

    int level, orientation, delta_qlog;



    for(level=0; level<s->spatial_decomposition_count; level++){

        for(orientation=level ? 1 : 0; orientation<4; orientation++){

            SubBand *b= &s->plane[0].band[level][orientation];

            DWTELEM *buf= b->buf;

            const int w= b->width;

            const int h= b->height;

            const int stride= b->stride;

            const int qlog= clip(2*QROOT + b->qlog, 0, QROOT*16);

            const int qmul= qexp[qlog&(QROOT-1)]<<(qlog>>QSHIFT);

            const int qdiv= (1<<16)/qmul;

            int x, y;

            if(orientation==0)

                decorrelate(s, b, buf, stride, 1, 0);

            for(y=0; y<h; y++)

                for(x=0; x<w; x++)

                    coef_sum+= abs(buf[x+y*stride]) * qdiv >> 16;

            if(orientation==0)

                correlate(s, b, buf, stride, 1, 0);

        }

    }



    /* ugly, ratecontrol just takes a sqrt again */

    coef_sum = (uint64_t)coef_sum * coef_sum >> 16;

    assert(coef_sum < INT_MAX);



    if(pict->pict_type == I_TYPE){

        s->m.current_picture.mb_var_sum= coef_sum;

        s->m.current_picture.mc_mb_var_sum= 0;

    }else{

        s->m.current_picture.mc_mb_var_sum= coef_sum;

        s->m.current_picture.mb_var_sum= 0;

    }



    pict->quality= ff_rate_estimate_qscale(&s->m, 1);

    if (pict->quality < 0)

        return -1;

    s->lambda= pict->quality * 3/2;

    delta_qlog= qscale2qlog(pict->quality) - s->qlog;

    s->qlog+= delta_qlog;

    return delta_qlog;

}
