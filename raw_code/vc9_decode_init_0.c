static int vc9_decode_init(AVCodecContext *avctx)

{

    VC9Context *v = avctx->priv_data;

    GetBitContext gb;



    if (!avctx->extradata_size || !avctx->extradata) return -1;

    avctx->pix_fmt = PIX_FMT_YUV420P;

    v->avctx = avctx;



    if (init_common(v) < 0) return -1;



    if (avctx->codec_id == CODEC_ID_WMV3)

    {

        int count = 0;



	// looks like WMV3 has a sequence header stored in the extradata

	// advanced sequence header may be before the first frame

	// the last byte of the extradata is a version number, 1 for the

	// samples we can decode



	init_get_bits(&gb, avctx->extradata, avctx->extradata_size);

	

	decode_sequence_header(avctx, &gb);



	count = avctx->extradata_size*8 - get_bits_count(&gb);

	if (count>0)

	{

    	    av_log(avctx, AV_LOG_INFO, "Extra data: %i bits left, value: %X\n",

               count, get_bits(&gb, count));

	}

	else

	{

    	    av_log(avctx, AV_LOG_INFO, "Read %i bits in overflow\n", -count);

	}

    }



    /* Done with header parsing */

    //FIXME I feel like this is wrong

    v->width_mb = (avctx->coded_width+15)>>4;

    v->height_mb = (avctx->coded_height+15)>>4;



    /* Allocate mb bitplanes */

    v->mv_type_mb_plane = (uint8_t *)av_malloc(v->width_mb*v->height_mb);

    if (!v->mv_type_mb_plane) return -1;

    v->skip_mb_plane = (uint8_t *)av_malloc(v->width_mb*v->height_mb);

    if (!v->skip_mb_plane) return -1;

    v->direct_mb_plane = (uint8_t *)av_malloc(v->width_mb*v->height_mb);

    if (!v->direct_mb_plane) return -1;



#if HAS_ADVANCED_PROFILE

    if (v->profile > PROFILE_MAIN)

    {

        v->over_flags_plane = (uint8_t *)av_malloc(v->width_mb*v->height_mb);

        if (!v->over_flags_plane) return -1;

        v->ac_pred_plane = (uint8_t *)av_malloc(v->width_mb*v->height_mb);

        if (!v->ac_pred_plane) return -1;

    }

#endif



    return 0;

}
