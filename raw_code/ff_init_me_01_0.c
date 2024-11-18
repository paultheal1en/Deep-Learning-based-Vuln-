void ff_init_me(MpegEncContext *s){

    MotionEstContext * const c= &s->me;

    c->avctx= s->avctx;



    ff_set_cmp(&s->dsp, s->dsp.me_pre_cmp, c->avctx->me_pre_cmp);

    ff_set_cmp(&s->dsp, s->dsp.me_cmp, c->avctx->me_cmp);

    ff_set_cmp(&s->dsp, s->dsp.me_sub_cmp, c->avctx->me_sub_cmp);

    ff_set_cmp(&s->dsp, s->dsp.mb_cmp, c->avctx->mb_cmp);

    

    c->flags    = get_flags(c, 0, c->avctx->me_cmp    &FF_CMP_CHROMA);

    c->sub_flags= get_flags(c, 0, c->avctx->me_sub_cmp&FF_CMP_CHROMA);

    c->mb_flags = get_flags(c, 0, c->avctx->mb_cmp    &FF_CMP_CHROMA);



/*FIXME s->no_rounding b_type*/

    if(s->flags&CODEC_FLAG_QPEL){

        c->sub_motion_search= qpel_motion_search;

        c->qpel_avg= s->dsp.avg_qpel_pixels_tab;

        if(s->no_rounding) c->qpel_put= s->dsp.put_no_rnd_qpel_pixels_tab;

        else               c->qpel_put= s->dsp.put_qpel_pixels_tab;

    }else{

        if(c->avctx->me_sub_cmp&FF_CMP_CHROMA)

            c->sub_motion_search= hpel_motion_search;

        else if(   c->avctx->me_sub_cmp == FF_CMP_SAD 

                && c->avctx->    me_cmp == FF_CMP_SAD 

                && c->avctx->    mb_cmp == FF_CMP_SAD)

            c->sub_motion_search= sad_hpel_motion_search; // 2050 vs. 2450 cycles

        else

            c->sub_motion_search= hpel_motion_search;

    }

    c->hpel_avg= s->dsp.avg_pixels_tab;

    if(s->no_rounding) c->hpel_put= s->dsp.put_no_rnd_pixels_tab;

    else               c->hpel_put= s->dsp.put_pixels_tab;



    if(s->linesize){

        c->stride  = s->linesize; 

        c->uvstride= s->uvlinesize;

    }else{

        c->stride  = 16*s->mb_width + 32;

        c->uvstride=  8*s->mb_width + 16;

    }



    // 8x8 fullpel search would need a 4x4 chroma compare, which we dont have yet, and even if we had the motion estimation code doesnt expect it

    if((c->avctx->me_cmp&FF_CMP_CHROMA) && !s->dsp.me_cmp[2]){

        s->dsp.me_cmp[2]= zero_cmp;

    }

    if((c->avctx->me_sub_cmp&FF_CMP_CHROMA) && !s->dsp.me_sub_cmp[2]){

        s->dsp.me_sub_cmp[2]= zero_cmp;

    }

    c->hpel_put[2][0]= c->hpel_put[2][1]=

    c->hpel_put[2][2]= c->hpel_put[2][3]= zero_hpel;



    c->temp= c->scratchpad;

}
