int ff_h263_decode_frame(AVCodecContext *avctx, 

                             void *data, int *data_size,

                             uint8_t *buf, int buf_size)

{

    MpegEncContext *s = avctx->priv_data;

    int ret;

    AVFrame *pict = data; 

    

#ifdef PRINT_FRAME_TIME

uint64_t time= rdtsc();

#endif

#ifdef DEBUG

    printf("*****frame %d size=%d\n", avctx->frame_number, buf_size);

    printf("bytes=%x %x %x %x\n", buf[0], buf[1], buf[2], buf[3]);

#endif

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



    if(s->flags&CODEC_FLAG_TRUNCATED){

        int next;

        

        if(s->codec_id==CODEC_ID_MPEG4){

            next= ff_mpeg4_find_frame_end(&s->parse_context, buf, buf_size);

        }else if(s->codec_id==CODEC_ID_H263){

            next= h263_find_frame_end(&s->parse_context, buf, buf_size);

        }else{

            av_log(s->avctx, AV_LOG_ERROR, "this codec doesnt support truncated bitstreams\n");

            return -1;

        }

        

        if( ff_combine_frame(&s->parse_context, next, &buf, &buf_size) < 0 )

            return buf_size;

    }



    

retry:

    

    if(s->bitstream_buffer_size && (s->divx_packed || buf_size<20)){ //divx 5.01+/xvid frame reorder

        init_get_bits(&s->gb, s->bitstream_buffer, s->bitstream_buffer_size*8);

    }else

        init_get_bits(&s->gb, buf, buf_size*8);

    s->bitstream_buffer_size=0;



    if (!s->context_initialized) {

        if (MPV_common_init(s) < 0) //we need the idct permutaton for reading a custom matrix

            return -1;

    }

    

    //we need to set current_picture_ptr before reading the header, otherwise we cant store anyting im there

    if(s->current_picture_ptr==NULL || s->current_picture_ptr->data[0]){

        int i= ff_find_unused_picture(s, 0);

        s->current_picture_ptr= &s->picture[i];

    }

      

    /* let's go :-) */

    if (s->msmpeg4_version==5) {

        ret= ff_wmv2_decode_picture_header(s);

    } else if (s->msmpeg4_version) {

        ret = msmpeg4_decode_picture_header(s);

    } else if (s->h263_pred) {

        if(s->avctx->extradata_size && s->picture_number==0){

            GetBitContext gb;

            

            init_get_bits(&gb, s->avctx->extradata, s->avctx->extradata_size*8);

            ret = ff_mpeg4_decode_picture_header(s, &gb);

        }

        ret = ff_mpeg4_decode_picture_header(s, &s->gb);



        if(s->flags& CODEC_FLAG_LOW_DELAY)

            s->low_delay=1;

    } else if (s->codec_id == CODEC_ID_H263I) {

        ret = intel_h263_decode_picture_header(s);

    } else if (s->h263_flv) {

        ret = flv_h263_decode_picture_header(s);

    } else {

        ret = h263_decode_picture_header(s);

    }

    

    if(ret==FRAME_SKIPED) return get_consumed_bytes(s, buf_size);



    /* skip if the header was thrashed */

    if (ret < 0){

        av_log(s->avctx, AV_LOG_ERROR, "header damaged\n");

        return -1;

    }

    

    avctx->has_b_frames= !s->low_delay;

    

    if(s->xvid_build==0 && s->divx_version==0 && s->lavc_build==0){

        if(s->avctx->stream_codec_tag == ff_get_fourcc("XVID") || 

           s->avctx->codec_tag == ff_get_fourcc("XVID") || s->avctx->codec_tag == ff_get_fourcc("XVIX"))

            s->xvid_build= -1;

#if 0

        if(s->avctx->codec_tag == ff_get_fourcc("DIVX") && s->vo_type==0 && s->vol_control_parameters==1

           && s->padding_bug_score > 0 && s->low_delay) // XVID with modified fourcc 

            s->xvid_build= -1;

#endif

    }



    if(s->xvid_build==0 && s->divx_version==0 && s->lavc_build==0){

        if(s->avctx->codec_tag == ff_get_fourcc("DIVX") && s->vo_type==0 && s->vol_control_parameters==0)

            s->divx_version= 400; //divx 4

    }



    if(s->workaround_bugs&FF_BUG_AUTODETECT){

        s->workaround_bugs &= ~FF_BUG_NO_PADDING;

        

        if(s->padding_bug_score > -2 && !s->data_partitioning && (s->divx_version || !s->resync_marker))

            s->workaround_bugs |=  FF_BUG_NO_PADDING;



        if(s->avctx->codec_tag == ff_get_fourcc("XVIX")) 

            s->workaround_bugs|= FF_BUG_XVID_ILACE;



        if(s->avctx->codec_tag == ff_get_fourcc("UMP4")){

            s->workaround_bugs|= FF_BUG_UMP4;

        }



        if(s->divx_version>=500){

            s->workaround_bugs|= FF_BUG_QPEL_CHROMA;

        }



        if(s->divx_version>502){

            s->workaround_bugs|= FF_BUG_QPEL_CHROMA2;

        }



        if(s->xvid_build && s->xvid_build<=3)

            s->padding_bug_score= 256*256*256*64;

        

        if(s->xvid_build && s->xvid_build<=1)

            s->workaround_bugs|= FF_BUG_QPEL_CHROMA;



        if(s->xvid_build && s->xvid_build<=12)

            s->workaround_bugs|= FF_BUG_EDGE;



        if(s->xvid_build && s->xvid_build<=32)

            s->workaround_bugs|= FF_BUG_DC_CLIP;



#define SET_QPEL_FUNC(postfix1, postfix2) \

    s->dsp.put_ ## postfix1 = ff_put_ ## postfix2;\

    s->dsp.put_no_rnd_ ## postfix1 = ff_put_no_rnd_ ## postfix2;\

    s->dsp.avg_ ## postfix1 = ff_avg_ ## postfix2;



        if(s->lavc_build && s->lavc_build<4653)

            s->workaround_bugs|= FF_BUG_STD_QPEL;

        

        if(s->lavc_build && s->lavc_build<4655)

            s->workaround_bugs|= FF_BUG_DIRECT_BLOCKSIZE;



        if(s->lavc_build && s->lavc_build<4670){

            s->workaround_bugs|= FF_BUG_EDGE;

        }

        

        if(s->lavc_build && s->lavc_build<=4712)

            s->workaround_bugs|= FF_BUG_DC_CLIP;



        if(s->divx_version)

            s->workaround_bugs|= FF_BUG_DIRECT_BLOCKSIZE;

//printf("padding_bug_score: %d\n", s->padding_bug_score);

        if(s->divx_version==501 && s->divx_build==20020416)

            s->padding_bug_score= 256*256*256*64;



        if(s->divx_version && s->divx_version<500){

            s->workaround_bugs|= FF_BUG_EDGE;

        }

        

        if(s->divx_version)

            s->workaround_bugs|= FF_BUG_HPEL_CHROMA;

#if 0

        if(s->divx_version==500)

            s->padding_bug_score= 256*256*256*64;



        /* very ugly XVID padding bug detection FIXME/XXX solve this differently

         * lets hope this at least works

         */

        if(   s->resync_marker==0 && s->data_partitioning==0 && s->divx_version==0

           && s->codec_id==CODEC_ID_MPEG4 && s->vo_type==0)

            s->workaround_bugs|= FF_BUG_NO_PADDING;

        

        if(s->lavc_build && s->lavc_build<4609) //FIXME not sure about the version num but a 4609 file seems ok

            s->workaround_bugs|= FF_BUG_NO_PADDING;

#endif

    }

    

    if(s->workaround_bugs& FF_BUG_STD_QPEL){

        SET_QPEL_FUNC(qpel_pixels_tab[0][ 5], qpel16_mc11_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][ 7], qpel16_mc31_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][ 9], qpel16_mc12_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][11], qpel16_mc32_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][13], qpel16_mc13_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[0][15], qpel16_mc33_old_c)



        SET_QPEL_FUNC(qpel_pixels_tab[1][ 5], qpel8_mc11_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][ 7], qpel8_mc31_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][ 9], qpel8_mc12_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][11], qpel8_mc32_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][13], qpel8_mc13_old_c)

        SET_QPEL_FUNC(qpel_pixels_tab[1][15], qpel8_mc33_old_c)

    }



    if(avctx->debug & FF_DEBUG_BUGS)

        av_log(s->avctx, AV_LOG_DEBUG, "bugs: %X lavc_build:%d xvid_build:%d divx_version:%d divx_build:%d %s\n", 

               s->workaround_bugs, s->lavc_build, s->xvid_build, s->divx_version, s->divx_build,

               s->divx_packed ? "p" : "");

    

#if 0 // dump bits per frame / qp / complexity

{

    static FILE *f=NULL;

    if(!f) f=fopen("rate_qp_cplx.txt", "w");

    fprintf(f, "%d %d %f\n", buf_size, s->qscale, buf_size*(double)s->qscale);

}

#endif

       

        /* After H263 & mpeg4 header decode we have the height, width,*/

        /* and other parameters. So then we could init the picture   */

        /* FIXME: By the way H263 decoder is evolving it should have */

        /* an H263EncContext                                         */

    

    if (   s->width != avctx->width || s->height != avctx->height) {

        /* H.263 could change picture size any time */

        ParseContext pc= s->parse_context; //FIXME move these demuxng hack to avformat

        s->parse_context.buffer=0;

        MPV_common_end(s);

        s->parse_context= pc;

    }

    if (!s->context_initialized) {

        avctx->width = s->width;

        avctx->height = s->height;



        goto retry;

    }



    if((s->codec_id==CODEC_ID_H263 || s->codec_id==CODEC_ID_H263P))

        s->gob_index = ff_h263_get_gob_height(s);

    

    // for hurry_up==5

    s->current_picture.pict_type= s->pict_type;

    s->current_picture.key_frame= s->pict_type == I_TYPE;



    /* skip b frames if we dont have reference frames */

    if(s->last_picture_ptr==NULL && s->pict_type==B_TYPE) return get_consumed_bytes(s, buf_size);

    /* skip b frames if we are in a hurry */

    if(avctx->hurry_up && s->pict_type==B_TYPE) return get_consumed_bytes(s, buf_size);

    /* skip everything if we are in a hurry>=5 */

    if(avctx->hurry_up>=5) return get_consumed_bytes(s, buf_size);

    

    if(s->next_p_frame_damaged){

        if(s->pict_type==B_TYPE)

            return get_consumed_bytes(s, buf_size);

        else

            s->next_p_frame_damaged=0;

    }



    if(MPV_frame_start(s, avctx) < 0)

        return -1;



#ifdef DEBUG

    printf("qscale=%d\n", s->qscale);

#endif



    ff_er_frame_start(s);

    

    //the second part of the wmv2 header contains the MB skip bits which are stored in current_picture->mb_type

    //which isnt available before MPV_frame_start()

    if (s->msmpeg4_version==5){

        if(ff_wmv2_decode_secondary_picture_header(s) < 0)

            return -1;

    }



    /* decode each macroblock */

    s->mb_x=0; 

    s->mb_y=0;

    

    decode_slice(s);

    while(s->mb_y<s->mb_height){

        if(s->msmpeg4_version){

            if(s->mb_x!=0 || (s->mb_y%s->slice_height)!=0 || get_bits_count(&s->gb) > s->gb.size_in_bits)

                break;

        }else{

            if(ff_h263_resync(s)<0)

                break;

        }

        

        if(s->msmpeg4_version<4 && s->h263_pred)

            ff_mpeg4_clean_buffers(s);



        decode_slice(s);

    }



    if (s->h263_msmpeg4 && s->msmpeg4_version<4 && s->pict_type==I_TYPE)

        if(msmpeg4_decode_ext_header(s, buf_size) < 0){

            s->error_status_table[s->mb_num-1]= AC_ERROR|DC_ERROR|MV_ERROR;

        }

    

    /* divx 5.01+ bistream reorder stuff */

    if(s->codec_id==CODEC_ID_MPEG4 && s->bitstream_buffer_size==0 && s->divx_packed){

        int current_pos= get_bits_count(&s->gb)>>3;

        int startcode_found=0;



        if(   buf_size - current_pos > 5 

           && buf_size - current_pos < BITSTREAM_BUFFER_SIZE){

            int i;

            for(i=current_pos; i<buf_size-3; i++){

                if(buf[i]==0 && buf[i+1]==0 && buf[i+2]==1 && buf[i+3]==0xB6){

                    startcode_found=1;

                    break;

                }

            }

        }

        if(s->gb.buffer == s->bitstream_buffer && buf_size>20){ //xvid style

            startcode_found=1;

            current_pos=0;

        }



        if(startcode_found){

            memcpy(s->bitstream_buffer, buf + current_pos, buf_size - current_pos);

            s->bitstream_buffer_size= buf_size - current_pos;

        }

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

#ifdef PRINT_FRAME_TIME

printf("%Ld\n", rdtsc()-time);

#endif



    return get_consumed_bytes(s, buf_size);

}
