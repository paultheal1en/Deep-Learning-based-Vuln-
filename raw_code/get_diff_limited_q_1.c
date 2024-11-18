static double get_diff_limited_q(MpegEncContext *s, RateControlEntry *rce, double q){

    RateControlContext *rcc= &s->rc_context;

    AVCodecContext *a= s->avctx;

    const int pict_type= rce->new_pict_type;

    const double last_p_q    = rcc->last_qscale_for[P_TYPE];

    const double last_non_b_q= rcc->last_qscale_for[rcc->last_non_b_pict_type];



    if     (pict_type==I_TYPE && (a->i_quant_factor>0.0 || rcc->last_non_b_pict_type==P_TYPE))

        q= last_p_q    *FFABS(a->i_quant_factor) + a->i_quant_offset;

    else if(pict_type==B_TYPE && a->b_quant_factor>0.0)

        q= last_non_b_q*    a->b_quant_factor  + a->b_quant_offset;




    /* last qscale / qdiff stuff */

    if(rcc->last_non_b_pict_type==pict_type || pict_type!=I_TYPE){

        double last_q= rcc->last_qscale_for[pict_type];

        const int maxdiff= FF_QP2LAMBDA * a->max_qdiff;



        if     (q > last_q + maxdiff) q= last_q + maxdiff;

        else if(q < last_q - maxdiff) q= last_q - maxdiff;

    }



    rcc->last_qscale_for[pict_type]= q; //Note we cannot do that after blurring



    if(pict_type!=B_TYPE)

        rcc->last_non_b_pict_type= pict_type;



    return q;

}