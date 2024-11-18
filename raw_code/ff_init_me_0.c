int ff_init_me(MpegEncContext *s){

    MotionEstContext * const c= &s->me;

    int cache_size= FFMIN(ME_MAP_SIZE>>ME_MAP_SHIFT, 1<<ME_MAP_SHIFT);

    int dia_size= FFMAX(FFABS(s->avctx->dia_size)&255, FFABS(s->avctx->pre_dia_size)&255);



    if(FFMIN(s->avctx->dia_size, s->avctx->pre_dia_size) < -ME_MAP_SIZE){

        av_log(s->avctx, AV_LOG_ERROR, "ME_MAP size is too small for SAB diamond\n");

        return -1;

    }

    //special case of snow is needed because snow uses its own iterative ME code

    if(s->me_method!=ME_ZERO && s->me_method!=ME_EPZS && s->me_method!=ME_X1 && s->avctx->codec_id != AV_CODEC_ID_SNOW){

        av_log(s->avctx, AV_LOG_ERROR, "me_method is only allowed to be set to zero and epzs; for hex,umh,full and others see dia_size\n");

        return -1;

    }



    c->avctx= s->avctx;



    if(cache_size < 2*dia_size && !c->stride){

        av_log(s->avctx, AV_LOG_INFO, "ME_MAP size may be a little small for the selected diamond size\n");

    }



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



    /* 8x8 fullpel search would need a 4x4 chroma compare, which we do

     * not have yet, and even if we had, the motion estimation code

     * does not expect it. */

    if(s->codec_id != AV_CODEC_ID_SNOW){

        if((c->avctx->me_cmp&FF_CMP_CHROMA)/* && !s->dsp.me_cmp[2]*/){

            s->dsp.me_cmp[2]= zero_cmp;

        }

        if((c->avctx->me_sub_cmp&FF_CMP_CHROMA) && !s->dsp.me_sub_cmp[2]){

            s->dsp.me_sub_cmp[2]= zero_cmp;

        }

        c->hpel_put[2][0]= c->hpel_put[2][1]=

        c->hpel_put[2][2]= c->hpel_put[2][3]= zero_hpel;

    }



    if(s->codec_id == AV_CODEC_ID_H261){

        c->sub_motion_search= no_sub_motion_search;

    }



    return 0;

}
