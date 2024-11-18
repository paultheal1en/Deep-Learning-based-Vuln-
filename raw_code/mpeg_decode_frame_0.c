static int mpeg_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             uint8_t *buf, int buf_size)

{

    Mpeg1Context *s = avctx->priv_data;

    const uint8_t *buf_end;

    const uint8_t *buf_ptr;

    uint32_t start_code;

    int ret, input_size;

    AVFrame *picture = data;

    MpegEncContext *s2 = &s->mpeg_enc_ctx;

    dprintf(avctx, "fill_buffer\n");



    if (buf_size == 0 || (buf_size == 4 && AV_RB32(buf) == SEQ_END_CODE)) {

        /* special case for last picture */

        if (s2->low_delay==0 && s2->next_picture_ptr) {

            *picture= *(AVFrame*)s2->next_picture_ptr;

            s2->next_picture_ptr= NULL;



            *data_size = sizeof(AVFrame);

        }

        return 0;

    }



    if(s2->flags&CODEC_FLAG_TRUNCATED){

        int next= ff_mpeg1_find_frame_end(&s2->parse_context, buf, buf_size);



        if( ff_combine_frame(&s2->parse_context, next, (const uint8_t **)&buf, &buf_size) < 0 )

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



    if(s->mpeg_enc_ctx_allocated==0 && avctx->codec_tag == ff_get_fourcc("VCR2"))

        vcr2_init_sequence(avctx);



    s->slice_count= 0;



    for(;;) {

        /* find start next code */

        start_code = -1;

        buf_ptr = ff_find_start_code(buf_ptr,buf_end, &start_code);

        if (start_code > 0x1ff){

            if(s2->pict_type != B_TYPE || avctx->skip_frame <= AVDISCARD_DEFAULT){

                if(avctx->thread_count > 1){

                    int i;



                    avctx->execute(avctx, slice_decode_thread,  (void**)&(s2->thread_context[0]), NULL, s->slice_count);

                    for(i=0; i<s->slice_count; i++)

                        s2->error_count += s2->thread_context[i]->error_count;

                }

                if (slice_end(avctx, picture)) {

                    if(s2->last_picture_ptr || s2->low_delay) //FIXME merge with the stuff in mpeg_decode_slice

                        *data_size = sizeof(AVPicture);

                }

            }

            return FFMAX(0, buf_ptr - buf - s2->parse_context.last_index);

        }



        input_size = buf_end - buf_ptr;



        if(avctx->debug & FF_DEBUG_STARTCODE){

            av_log(avctx, AV_LOG_DEBUG, "%3X at %zd left %d\n", start_code, buf_ptr-buf, input_size);

        }



        /* prepare data for next start code */

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

        case GOP_START_CODE:

            s2->first_field=0;

            mpeg_decode_gop(avctx,

                                    buf_ptr, input_size);

            break;

        default:

            if (start_code >= SLICE_MIN_START_CODE &&

                start_code <= SLICE_MAX_START_CODE) {

                int mb_y= start_code - SLICE_MIN_START_CODE;



                if(s2->last_picture_ptr==NULL){

                /* Skip B-frames if we do not have reference frames. */

                    if(s2->pict_type==B_TYPE) break;

                /* Skip P-frames if we do not have reference frame no valid header. */

//                    if(s2->pict_type==P_TYPE && s2->first_field && !s2->first_slice) break;

                }

                /* Skip B-frames if we are in a hurry. */

                if(avctx->hurry_up && s2->pict_type==B_TYPE) break;

                if(  (avctx->skip_frame >= AVDISCARD_NONREF && s2->pict_type==B_TYPE)

                    ||(avctx->skip_frame >= AVDISCARD_NONKEY && s2->pict_type!=I_TYPE)

                    || avctx->skip_frame >= AVDISCARD_ALL)

                    break;

                /* Skip everything if we are in a hurry>=5. */

                if(avctx->hurry_up>=5) break;



                if (!s->mpeg_enc_ctx_allocated) break;



                if(s2->codec_id == CODEC_ID_MPEG2VIDEO){

                    if(mb_y < avctx->skip_top || mb_y >= s2->mb_height - avctx->skip_bottom)

                        break;

                }



                if(s2->first_slice){

                    s2->first_slice=0;

                            if(mpeg_field_start(s2) < 0)

                        return -1;

                    }

                if(!s2->current_picture_ptr){

                    av_log(avctx, AV_LOG_ERROR, "current_picture not initialized\n");

                    return -1;

                }



                if(avctx->thread_count > 1){

                    int threshold= (s2->mb_height*s->slice_count + avctx->thread_count/2) / avctx->thread_count;

                    if(threshold <= mb_y){

                        MpegEncContext *thread_context= s2->thread_context[s->slice_count];



                        thread_context->start_mb_y= mb_y;

                        thread_context->end_mb_y  = s2->mb_height;

                        if(s->slice_count){

                            s2->thread_context[s->slice_count-1]->end_mb_y= mb_y;

                            ff_update_duplicate_context(thread_context, s2);

                        }

                        init_get_bits(&thread_context->gb, buf_ptr, input_size*8);

                        s->slice_count++;

                    }

                    buf_ptr += 2; //FIXME add minimum num of bytes per slice

                }else{

                    ret = mpeg_decode_slice(s, mb_y, &buf_ptr, input_size);

                    emms_c();



                    if(ret < 0){

                        if(s2->resync_mb_x>=0 && s2->resync_mb_y>=0)

                            ff_er_add_slice(s2, s2->resync_mb_x, s2->resync_mb_y, s2->mb_x, s2->mb_y, AC_ERROR|DC_ERROR|MV_ERROR);

                    }else{

                        ff_er_add_slice(s2, s2->resync_mb_x, s2->resync_mb_y, s2->mb_x-1, s2->mb_y, AC_END|DC_END|MV_END);

                    }

                }

            }

            break;

        }

    }

}
