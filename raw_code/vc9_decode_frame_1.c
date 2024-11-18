static int vc9_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            uint8_t *buf, int buf_size)

{

    VC9Context *v = avctx->priv_data;

    MpegEncContext *s = &v->s;

    int ret = FRAME_SKIPED, len, start_code;

    AVFrame *pict = data;

    uint8_t *tmp_buf;

    v->s.avctx = avctx;



    //buf_size = 0 -> last frame

    if (!buf_size) return 0;



    len = avpicture_get_size(avctx->pix_fmt, avctx->width,

                             avctx->height);

    tmp_buf = (uint8_t *)av_mallocz(len);

    avpicture_fill((AVPicture *)pict, tmp_buf, avctx->pix_fmt,

                   avctx->width, avctx->height);



    if (avctx->codec_id == CODEC_ID_VC9)

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



	    init_get_bits(gb, buf+i, (buf_size-i)*8);

	

	    switch(scs)

	    {

            case 0x0A: //Sequence End Code

                return 0;

            case 0x0B: //Slice Start Code

                av_log(avctx, AV_LOG_ERROR, "Slice coding not supported\n");

                return -1;

            case 0x0C: //Field start code

                av_log(avctx, AV_LOG_ERROR, "Interlaced coding not supported\n");

                return -1;

            case 0x0D: //Frame start code

                break;

            case 0x0E: //Entry point Start Code

                if (v->profile <= MAIN_PROFILE)

                    av_log(avctx, AV_LOG_ERROR,

                           "Found an entry point in profile %i\n", v->profile);

                advanced_entry_point_process(avctx, gb);

                break;

            case 0x0F: //Sequence header Start Code

                decode_sequence_header(avctx, gb);

                break;

            default:

                av_log(avctx, AV_LOG_ERROR,

                       "Unsupported IDU suffix %lX\n", scs);

            }

	    

	    i += get_bits_count(gb)*8;

	}

#else

	av_abort();

#endif

    }

    else

        init_get_bits(&v->s.gb, buf, buf_size*8);



    s->flags= avctx->flags;

    s->flags2= avctx->flags2;



    /* no supplementary picture */

    if (buf_size == 0) {

        /* special case for last picture */

        if (s->low_delay==0 && s->next_picture_ptr) {

            *pict= *(AVFrame*)s->next_picture_ptr;

            s->next_picture_ptr= NULL;



            *data_size = sizeof(AVFrame);

        }



        return 0;

    }



    //No IDU - we mimic ff_h263_decode_frame

    s->bitstream_buffer_size=0;

        

    if (!s->context_initialized) {

        if (MPV_common_init(s) < 0) //we need the idct permutaton for reading a custom matrix

            return -1;

    }

    

    //we need to set current_picture_ptr before reading the header, otherwise we cant store anyting im there

    if(s->current_picture_ptr==NULL || s->current_picture_ptr->data[0]){

        s->current_picture_ptr= &s->picture[ff_find_unused_picture(s, 0)];

    }

#if HAS_ADVANCED_PROFILE

    if (v->profile > PROFILE_MAIN)

        ret= advanced_decode_picture_primary_header(v);

    else

#endif

        ret= standard_decode_picture_primary_header(v);

    if (ret == FRAME_SKIPED) return buf_size;

    /* skip if the header was thrashed */

    if (ret < 0){

        av_log(s->avctx, AV_LOG_ERROR, "header damaged\n");

        return -1;

    }



    //No bug workaround yet, no DCT conformance



    //WMV9 does have resized images

    if (v->profile <= PROFILE_MAIN && v->multires){

        //Parse context stuff in here, don't know how appliable it is

    }

    //Not sure about context initialization



    // for hurry_up==5

    s->current_picture.pict_type= s->pict_type;

    s->current_picture.key_frame= s->pict_type == I_TYPE;



    /* skip b frames if we dont have reference frames */

    if(s->last_picture_ptr==NULL && (s->pict_type==B_TYPE || s->dropable))

        return buf_size; //FIXME simulating all buffer consumed

    /* skip b frames if we are in a hurry */

    if(avctx->hurry_up && s->pict_type==B_TYPE)

        return buf_size; //FIXME simulating all buffer consumed

    /* skip everything if we are in a hurry>=5 */

    if(avctx->hurry_up>=5)

        return buf_size; //FIXME simulating all buffer consumed

    

    if(s->next_p_frame_damaged){

        if(s->pict_type==B_TYPE)

            return buf_size; //FIXME simulating all buffer consumed

        else

            s->next_p_frame_damaged=0;

    }



    if(MPV_frame_start(s, avctx) < 0)

        return -1;



    ff_er_frame_start(s);



    //wmv9 may or may not have skip bits

#if HAS_ADVANCED_PROFILE

    if (v->profile > PROFILE_MAIN)

        ret= advanced_decode_picture_secondary_header(v);

    else

#endif

        ret = standard_decode_picture_secondary_header(v);

    if (ret<0) return FRAME_SKIPED; //FIXME Non fatal for now



    //We consider the image coded in only one slice

#if HAS_ADVANCED_PROFILE

    if (v->profile > PROFILE_MAIN)

    {

        switch(s->pict_type)

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

        switch(s->pict_type)

        {

            case I_TYPE: ret = standard_decode_i_mbs(v); break;

            case P_TYPE: ret = decode_p_mbs(v); break;

            case B_TYPE:

            case BI_TYPE: ret = decode_b_mbs(v); break;

            default: ret = FRAME_SKIPED;

        }

        if (ret == FRAME_SKIPED) return buf_size;

    }



    ff_er_frame_end(s);



    MPV_frame_end(s);



    assert(s->current_picture.pict_type == s->current_picture_ptr->pict_type);

    assert(s->current_picture.pict_type == s->pict_type);

    if(s->pict_type==B_TYPE || s->low_delay){

        *pict= *(AVFrame*)&s->current_picture;

        ff_print_debug_info(s, pict);

    } else {

        *pict= *(AVFrame*)&s->last_picture;

        if(pict)

            ff_print_debug_info(s, pict);

    }



    /* Return the Picture timestamp as the frame number */

    /* we substract 1 because it is added on utils.c    */

    avctx->frame_number = s->picture_number - 1;



    /* dont output the last pic after seeking */

    if(s->last_picture_ptr || s->low_delay)

        *data_size = sizeof(AVFrame);



    av_log(avctx, AV_LOG_DEBUG, "Consumed %i/%i bits\n",

           get_bits_count(&s->gb), buf_size*8);



    /* Fake consumption of all data */

    *data_size = len;

    return buf_size; //Number of bytes consumed

}
