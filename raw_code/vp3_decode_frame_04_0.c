static int vp3_decode_frame(AVCodecContext *avctx, 

                            void *data, int *data_size,

                            uint8_t *buf, int buf_size)

{

    Vp3DecodeContext *s = avctx->priv_data;

    GetBitContext gb;

    static int counter = 0;



    init_get_bits(&gb, buf, buf_size * 8);

    

    if (s->theora && get_bits1(&gb))

    {

	int ptype = get_bits(&gb, 7);



	skip_bits(&gb, 6*8); /* "theora" */

	

	switch(ptype)

	{

	    case 1:

		theora_decode_comments(avctx, gb);

		break;

	    case 2:

		theora_decode_tables(avctx, gb);

    		init_dequantizer(s);

		break;

	    default:

		av_log(avctx, AV_LOG_ERROR, "Unknown Theora config packet: %d\n", ptype);

	}

	return buf_size;

    }



    s->keyframe = !get_bits1(&gb);

    if (!s->theora)

	skip_bits(&gb, 1);

    s->last_quality_index = s->quality_index;

    s->quality_index = get_bits(&gb, 6);

    if (s->theora >= 0x030200)

        skip_bits1(&gb);



    if (s->avctx->debug & FF_DEBUG_PICT_INFO)

	av_log(s->avctx, AV_LOG_INFO, " VP3 %sframe #%d: Q index = %d\n",

	    s->keyframe?"key":"", counter, s->quality_index);

    counter++;



    if (s->quality_index != s->last_quality_index)

        init_dequantizer(s);



    if (s->keyframe) {

	if (!s->theora)

	{

	    skip_bits(&gb, 4); /* width code */

	    skip_bits(&gb, 4); /* height code */

	    if (s->version)

	    {

		s->version = get_bits(&gb, 5);

		if (counter == 1)

		    av_log(s->avctx, AV_LOG_DEBUG, "VP version: %d\n", s->version);

	    }

	}

	if (s->version || s->theora)

	{

    	    if (get_bits1(&gb))

    	        av_log(s->avctx, AV_LOG_ERROR, "Warning, unsupported keyframe coding type?!\n");

	    skip_bits(&gb, 2); /* reserved? */

	}



        if (s->last_frame.data[0] == s->golden_frame.data[0]) {

            if (s->golden_frame.data[0])

                avctx->release_buffer(avctx, &s->golden_frame);

            s->last_frame= s->golden_frame; /* ensure that we catch any access to this released frame */

        } else {

            if (s->golden_frame.data[0])

                avctx->release_buffer(avctx, &s->golden_frame);

            if (s->last_frame.data[0])

                avctx->release_buffer(avctx, &s->last_frame);

        }



        s->golden_frame.reference = 3;

        if(avctx->get_buffer(avctx, &s->golden_frame) < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "vp3: get_buffer() failed\n");

            return -1;

        }



        /* golden frame is also the current frame */

        memcpy(&s->current_frame, &s->golden_frame, sizeof(AVFrame));



        /* time to figure out pixel addresses? */

        if (!s->pixel_addresses_inited)

	{

	    if (!s->flipped_image)

        	vp3_calculate_pixel_addresses(s);

	    else

		theora_calculate_pixel_addresses(s);

	}

    } else {

        /* allocate a new current frame */

        s->current_frame.reference = 3;

        if(avctx->get_buffer(avctx, &s->current_frame) < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "vp3: get_buffer() failed\n");

            return -1;

        }

    }



    s->current_frame.qscale_table= s->qscale_table; //FIXME allocate individual tables per AVFrame

    s->current_frame.qstride= 0;



    init_frame(s, &gb);



#if KEYFRAMES_ONLY

if (!s->keyframe) {



    memcpy(s->current_frame.data[0], s->golden_frame.data[0],

        s->current_frame.linesize[0] * s->height);

    memcpy(s->current_frame.data[1], s->golden_frame.data[1],

        s->current_frame.linesize[1] * s->height / 2);

    memcpy(s->current_frame.data[2], s->golden_frame.data[2],

        s->current_frame.linesize[2] * s->height / 2);



} else {

#endif



    if (unpack_superblocks(s, &gb) ||

        unpack_modes(s, &gb) ||

        unpack_vectors(s, &gb) ||

        unpack_dct_coeffs(s, &gb)) {



        av_log(s->avctx, AV_LOG_ERROR, "  vp3: could not decode frame\n");

        return -1;

    }



    reverse_dc_prediction(s, 0, s->fragment_width, s->fragment_height);

    render_fragments(s, 0, s->width, s->height, 0);

//    apply_loop_filter(s);



    if ((avctx->flags & CODEC_FLAG_GRAY) == 0) {

        reverse_dc_prediction(s, s->u_fragment_start,

            s->fragment_width / 2, s->fragment_height / 2);

        reverse_dc_prediction(s, s->v_fragment_start,

            s->fragment_width / 2, s->fragment_height / 2);

        render_fragments(s, s->u_fragment_start, s->width / 2, s->height / 2, 1);

        render_fragments(s, s->v_fragment_start, s->width / 2, s->height / 2, 2);

    } else {

        memset(s->current_frame.data[1], 0x80, s->width * s->height / 4);

        memset(s->current_frame.data[2], 0x80, s->width * s->height / 4);

    }



#if KEYFRAMES_ONLY

}

#endif



    *data_size=sizeof(AVFrame);

    *(AVFrame*)data= s->current_frame;



    /* release the last frame, if it is allocated and if it is not the

     * golden frame */

    if ((s->last_frame.data[0]) &&

        (s->last_frame.data[0] != s->golden_frame.data[0]))

        avctx->release_buffer(avctx, &s->last_frame);



    /* shuffle frames (last = current) */

    memcpy(&s->last_frame, &s->current_frame, sizeof(AVFrame));

    s->current_frame.data[0]= NULL; /* ensure that we catch any access to this released frame */



    return buf_size;

}
