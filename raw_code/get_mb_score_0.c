static int inline get_mb_score(MpegEncContext * s, int mx, int my, int src_index,

                               int ref_index)

{

//    const int check_luma= s->dsp.me_sub_cmp != s->dsp.mb_cmp;

    MotionEstContext * const c= &s->me;

    const int size= 0;

    const int h= 16;

    const int penalty_factor= c->mb_penalty_factor;

    const int flags= c->mb_flags;

    const int qpel= flags & FLAG_QPEL;

    const int mask= 1+2*qpel;

    me_cmp_func cmp_sub, chroma_cmp_sub;

    int d;



    LOAD_COMMON

    

 //FIXME factorize



    cmp_sub= s->dsp.mb_cmp[size];

    chroma_cmp_sub= s->dsp.mb_cmp[size+1];

    

    assert(!c->skip);

    assert(c->avctx->me_sub_cmp != c->avctx->mb_cmp);



    d= cmp(s, mx>>(qpel+1), my>>(qpel+1), mx&mask, my&mask, size, h, ref_index, src_index, cmp_sub, chroma_cmp_sub, flags);

    //FIXME check cbp before adding penalty for (0,0) vector

    if(mx || my || size>0)

        d += (mv_penalty[mx - pred_x] + mv_penalty[my - pred_y])*penalty_factor;

        

    return d;

}
