static int mpeg_decode_frame(AVCodecContext *avctx, 

                             void *data, int *data_size,

                             uint8_t *buf, int buf_size)

{

    Mpeg1Context *s = avctx->priv_data;

    uint8_t *buf_end, *buf_ptr;

    int ret, start_code, input_size;

    AVFrame *picture = data;

    MpegEncContext *s2 = &s->mpeg_enc_ctx;

    dprintf("fill_buffer\n");



    *data_size = 0;



    /* special case for last picture */

    if (buf_size == 0) {

        if (s2->picture_number > 0) {

            *picture= *(AVFrame*)&s2->next_picture;



            *data_size = sizeof(AVFrame);

        }

        return 0;

    }



    if(s2->flags&CODEC_FLAG_TRUNCATED){

        int next;

        

        next= mpeg1_find_frame_end(s2, buf, buf_size);

        

        if( ff_combine_frame(s2, next, &buf, &buf_size) < 0 )

            return buf_size;

    }    

    

    buf_ptr = buf;

    buf_end = buf + buf_size;



#if 0    

    if (s->repeat_field % 2 == 1) { 

        s->repeat_field++;

        //fprintf(stderr,"\nRepeating last frame: %d -> %d! pict: %d %d", avctx->frame_number-1, avctx->frame_number,

        //        s2->picture_number, s->repeat_field);

        if (avctx->flags & CODEC_FLAG_REPEAT_FIELD) {

            *data_size = sizeof(AVPicture);

            goto the_end;

        }

    }

#endif

    for(;;) {

        /* find start next code */

        start_code = find_start_code(&buf_ptr, buf_end);

        if (start_code < 0){ 

            printf("missing end of picture\n");

            return FFMAX(1, buf_ptr - buf - s2->parse_context.last_index);

        }



                /* prepare data for next start code */

                input_size = buf_end - buf_ptr;

                switch(start_code) {

                case SEQ_START_CODE:

                    mpeg1_decode_sequence(avctx, buf_ptr, 

                                          input_size);

                    break;

                            

                case PICTURE_START_CODE:

                    /* we have a complete image : we try to decompress it */

                    mpeg1_decode_picture(avctx, 

                                         buf_ptr, input_size);

                    break;

                case EXT_START_CODE:

                    mpeg_decode_extension(avctx,

                                          buf_ptr, input_size);

                    break;

                case USER_START_CODE:

                    mpeg_decode_user_data(avctx, 

                                          buf_ptr, input_size);

                    break;

                default:

                    if (start_code >= SLICE_MIN_START_CODE &&

                        start_code <= SLICE_MAX_START_CODE) {

                        

                        /* skip b frames if we dont have reference frames */

                        if(s2->last_picture_ptr==NULL && s2->pict_type==B_TYPE) break;

                        /* skip b frames if we are in a hurry */

                        if(avctx->hurry_up && s2->pict_type==B_TYPE) break;

                        /* skip everything if we are in a hurry>=5 */

                        if(avctx->hurry_up>=5) break;

                        

                        if (!s->mpeg_enc_ctx_allocated) break;



                        ret = mpeg_decode_slice(avctx, picture,

                                                start_code, &buf_ptr, input_size);



                        if (ret == DECODE_SLICE_EOP) {

                            if(s2->last_picture_ptr) //FIXME merge with the stuff in mpeg_decode_slice

                                *data_size = sizeof(AVPicture);

                            return FFMAX(1, buf_ptr - buf - s2->parse_context.last_index);

                        }else if(ret < 0){

                            if(ret == DECODE_SLICE_ERROR)

                                ff_er_add_slice(s2, s2->resync_mb_x, s2->resync_mb_y, s2->mb_x, s2->mb_y, AC_ERROR|DC_ERROR|MV_ERROR);

                                

                            fprintf(stderr,"Error while decoding slice\n");

			    if(ret==DECODE_SLICE_FATAL_ERROR) return -1;

                        }

                    }

                    break;

                }

    }

}
