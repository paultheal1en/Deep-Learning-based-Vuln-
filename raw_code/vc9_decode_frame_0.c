static int vc9_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            uint8_t *buf, int buf_size)

{

    VC9Context *v = avctx->priv_data;

    int ret = FRAME_SKIPED, len, start_code;

    AVFrame *pict = data;

    uint8_t *tmp_buf;

    v->avctx = avctx;



    //buf_size = 0 -> last frame

    if (!buf_size) return 0;



    len = avpicture_get_size(avctx->pix_fmt, avctx->width,

                             avctx->height);

    tmp_buf = (uint8_t *)av_mallocz(len);

    avpicture_fill((AVPicture *)pict, tmp_buf, avctx->pix_fmt,

                   avctx->width, avctx->height);



    if (avctx->codec_id == CODEC_ID_WMV3)

    {

	init_get_bits(&v->gb, buf, buf_size*8);

	av_log(avctx, AV_LOG_INFO, "Frame: %i bits to decode\n", buf_size*8);

        

#if HAS_ADVANCED_PROFILE

	if (v->profile > PROFILE_MAIN)

	{

    	    if (advanced_decode_picture_header(v) == FRAME_SKIPED) return buf_size;

    	    switch(v->pict_type)

    	    {

    		case I_TYPE: ret = advanced_decode_i_mbs(v); break;

	        case P_TYPE: ret = decode_p_mbs(v); break;

    		case B_TYPE:

    		case BI_TYPE: ret = decode_b_mbs(v); break;

    		default: ret = FRAME_SKIPED;

    	    }

    	    if (ret == FRAME_SKIPED) return buf_size; //We ignore for now failures

	}

	else

#endif

	{

    	    if (standard_decode_picture_header(v) == FRAME_SKIPED) return buf_size;

    	    switch(v->pict_type)

	    {

    		case I_TYPE: ret = standard_decode_i_mbs(v); break;

    		case P_TYPE: ret = decode_p_mbs(v); break;

    		case B_TYPE:

    		case BI_TYPE: ret = decode_b_mbs(v); break;

    		default: ret = FRAME_SKIPED;

    	    }

    	    if (ret == FRAME_SKIPED) return buf_size;

	}



	/* Size of the output data = image */

	av_log(avctx, AV_LOG_DEBUG, "Consumed %i/%i bits\n",

           get_bits_count(&v->gb), buf_size*8);

    }

    else

    {

#if 0

	// search for IDU's

	// FIXME

	uint32_t scp = 0;

	int scs = 0, i = 0;



	while (i < buf_size)

	{

	    for (; i < buf_size && scp != 0x000001; i++)

		scp = ((scp<<8)|buf[i])&0xffffff;



	    if (scp != 0x000001)

		break; // eof ?

	

	    scs = buf[i++];	



	    init_get_bits(&v->gb, buf+i, (buf_size-i)*8);

	

	    switch(scs)

	    {

		case 0xf:

		    decode_sequence_header(avctx, &v->gb);

		    break;

		// to be finished

	    }

	    

	    i += get_bits_count(&v->gb)*8;

	}

#else

	av_abort();

#endif

    }



    *data_size = len;



    /* Fake consumption of all data */

    return buf_size; //Number of bytes consumed

}
