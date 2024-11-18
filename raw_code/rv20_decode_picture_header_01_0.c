static int rv20_decode_picture_header(MpegEncContext *s)

{

    int seq, mb_pos, i;



    i= get_bits(&s->gb, 2);

    switch(i){

    case 0: s->pict_type= I_TYPE; break;

    case 1: s->pict_type= I_TYPE; break; //hmm ...

    case 2: s->pict_type= P_TYPE; break;

    case 3: s->pict_type= B_TYPE; break;

    default: 

        av_log(s->avctx, AV_LOG_ERROR, "unknown frame type\n");

        return -1;

    }

    

    if (get_bits(&s->gb, 1)){

        av_log(s->avctx, AV_LOG_ERROR, "unknown bit set\n");

        return -1;

    }



    s->qscale = get_bits(&s->gb, 5);

    if(s->qscale==0){

        av_log(s->avctx, AV_LOG_ERROR, "error, qscale:0\n");

        return -1;

    }



    if(s->avctx->sub_id == 0x20200002)

        seq= get_bits(&s->gb, 16);

    else

        seq= get_bits(&s->gb, 8);



    for(i=0; i<6; i++){

        if(s->mb_width*s->mb_height < ff_mba_max[i]) break;

    }

    mb_pos= get_bits(&s->gb, ff_mba_length[i]);

    s->mb_x= mb_pos % s->mb_width;

    s->mb_y= mb_pos / s->mb_width;

    s->no_rounding= get_bits1(&s->gb);

    

    s->f_code = 1;

    s->unrestricted_mv = 1;

    s->h263_aic= s->pict_type == I_TYPE;

//    s->alt_inter_vlc=1;

//    s->obmc=1;

//    s->umvplus=1;

//    s->modified_quant=1;

    

    if(s->avctx->debug & FF_DEBUG_PICT_INFO){

            av_log(s->avctx, AV_LOG_INFO, "num:%5d x:%2d y:%2d type:%d qscale:%2d rnd:%d\n", 

                   seq, s->mb_x, s->mb_y, s->pict_type, s->qscale, s->no_rounding);

    }



    if (s->pict_type == B_TYPE){

        av_log(s->avctx, AV_LOG_ERROR, "b frame not supported\n");

        return -1;

    }



    return s->mb_width*s->mb_height - mb_pos;

}
