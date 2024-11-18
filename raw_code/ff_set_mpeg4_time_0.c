void ff_set_mpeg4_time(MpegEncContext * s){

    if(s->pict_type==AV_PICTURE_TYPE_B){

        ff_mpeg4_init_direct_mv(s);

    }else{

        s->last_time_base= s->time_base;

        s->time_base= s->time/s->avctx->time_base.den;

    }

}
