static int estimate_qp(MpegEncContext *s, int dry_run){

    if (s->next_lambda){

        s->current_picture_ptr->quality=

        s->current_picture.quality = s->next_lambda;

        if(!dry_run) s->next_lambda= 0;

    } else if (!s->fixed_qscale) {

        s->current_picture_ptr->quality=

        s->current_picture.quality = ff_rate_estimate_qscale(s, dry_run);

        if (s->current_picture.quality < 0)

            return -1;

    }



    if(s->adaptive_quant){

        switch(s->codec_id){

        case CODEC_ID_MPEG4:

            if (CONFIG_MPEG4_ENCODER)

                ff_clean_mpeg4_qscales(s);

            break;

        case CODEC_ID_H263:

        case CODEC_ID_H263P:

        case CODEC_ID_FLV1:

            if (CONFIG_H263_ENCODER||CONFIG_H263P_ENCODER||CONFIG_FLV_ENCODER)

                ff_clean_h263_qscales(s);

            break;

        }



        s->lambda= s->lambda_table[0];

        //FIXME broken

    }else

        s->lambda= s->current_picture.quality;

//printf("%d %d\n", s->avctx->global_quality, s->current_picture.quality);

    update_qscale(s);

    return 0;

}
