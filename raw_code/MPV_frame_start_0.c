int MPV_frame_start(MpegEncContext *s, AVCodecContext *avctx)

{

    int i;

    Picture *pic;

    s->mb_skipped = 0;



    assert(s->last_picture_ptr==NULL || s->out_format != FMT_H264 || s->codec_id == CODEC_ID_SVQ3);



    /* mark&release old frames */

    if (s->pict_type != FF_B_TYPE && s->last_picture_ptr && s->last_picture_ptr != s->next_picture_ptr && s->last_picture_ptr->data[0]) {

      if(s->out_format != FMT_H264 || s->codec_id == CODEC_ID_SVQ3){

          free_frame_buffer(s, s->last_picture_ptr);



        /* release forgotten pictures */

        /* if(mpeg124/h263) */

        if(!s->encoding){

            for(i=0; i<MAX_PICTURE_COUNT; i++){

                if(s->picture[i].data[0] && &s->picture[i] != s->next_picture_ptr && s->picture[i].reference){

                    av_log(avctx, AV_LOG_ERROR, "releasing zombie picture\n");

                    free_frame_buffer(s, &s->picture[i]);

                }

            }

        }

      }

    }

alloc:

    if(!s->encoding){

        /* release non reference frames */

        for(i=0; i<MAX_PICTURE_COUNT; i++){

            if(s->picture[i].data[0] && !s->picture[i].reference /*&& s->picture[i].type!=FF_BUFFER_TYPE_SHARED*/){

                free_frame_buffer(s, &s->picture[i]);

            }

        }



        if(s->current_picture_ptr && s->current_picture_ptr->data[0]==NULL)

            pic= s->current_picture_ptr; //we already have a unused image (maybe it was set before reading the header)

        else{

            i= ff_find_unused_picture(s, 0);

            pic= &s->picture[i];

        }



        pic->reference= 0;

        if (!s->dropable){

            if (s->codec_id == CODEC_ID_H264)

                pic->reference = s->picture_structure;

            else if (s->pict_type != FF_B_TYPE)

                pic->reference = 3;

        }



        pic->coded_picture_number= s->coded_picture_number++;



        if(ff_alloc_picture(s, pic, 0) < 0)

            return -1;



        s->current_picture_ptr= pic;

        s->current_picture_ptr->top_field_first= s->top_field_first; //FIXME use only the vars from current_pic

        s->current_picture_ptr->interlaced_frame= !s->progressive_frame && !s->progressive_sequence;

    }



    s->current_picture_ptr->pict_type= s->pict_type;

//    if(s->flags && CODEC_FLAG_QSCALE)

  //      s->current_picture_ptr->quality= s->new_picture_ptr->quality;

    s->current_picture_ptr->key_frame= s->pict_type == FF_I_TYPE;



    ff_copy_picture(&s->current_picture, s->current_picture_ptr);



    if (s->pict_type != FF_B_TYPE) {

        s->last_picture_ptr= s->next_picture_ptr;

        if(!s->dropable)

            s->next_picture_ptr= s->current_picture_ptr;

    }

/*    av_log(s->avctx, AV_LOG_DEBUG, "L%p N%p C%p L%p N%p C%p type:%d drop:%d\n", s->last_picture_ptr, s->next_picture_ptr,s->current_picture_ptr,

        s->last_picture_ptr    ? s->last_picture_ptr->data[0] : NULL,

        s->next_picture_ptr    ? s->next_picture_ptr->data[0] : NULL,

        s->current_picture_ptr ? s->current_picture_ptr->data[0] : NULL,

        s->pict_type, s->dropable);*/



    if(s->last_picture_ptr) ff_copy_picture(&s->last_picture, s->last_picture_ptr);

    if(s->next_picture_ptr) ff_copy_picture(&s->next_picture, s->next_picture_ptr);



    if(s->pict_type != FF_I_TYPE && (s->last_picture_ptr==NULL || s->last_picture_ptr->data[0]==NULL) && !s->dropable && s->codec_id != CODEC_ID_H264){

        av_log(avctx, AV_LOG_ERROR, "warning: first frame is no keyframe\n");

        assert(s->pict_type != FF_B_TYPE); //these should have been dropped if we don't have a reference

        goto alloc;

    }



    assert(s->pict_type == FF_I_TYPE || (s->last_picture_ptr && s->last_picture_ptr->data[0]));



    if(s->picture_structure!=PICT_FRAME && s->out_format != FMT_H264){

        int i;

        for(i=0; i<4; i++){

            if(s->picture_structure == PICT_BOTTOM_FIELD){

                 s->current_picture.data[i] += s->current_picture.linesize[i];

            }

            s->current_picture.linesize[i] *= 2;

            s->last_picture.linesize[i] *=2;

            s->next_picture.linesize[i] *=2;

        }

    }



    s->hurry_up= s->avctx->hurry_up;

    s->error_recognition= avctx->error_recognition;



    /* set dequantizer, we can't do it during init as it might change for mpeg4

       and we can't do it in the header decode as init is not called for mpeg4 there yet */

    if(s->mpeg_quant || s->codec_id == CODEC_ID_MPEG2VIDEO){

        s->dct_unquantize_intra = s->dct_unquantize_mpeg2_intra;

        s->dct_unquantize_inter = s->dct_unquantize_mpeg2_inter;

    }else if(s->out_format == FMT_H263 || s->out_format == FMT_H261){

        s->dct_unquantize_intra = s->dct_unquantize_h263_intra;

        s->dct_unquantize_inter = s->dct_unquantize_h263_inter;

    }else{

        s->dct_unquantize_intra = s->dct_unquantize_mpeg1_intra;

        s->dct_unquantize_inter = s->dct_unquantize_mpeg1_inter;

    }



    if(s->dct_error_sum){

        assert(s->avctx->noise_reduction && s->encoding);



        update_noise_reduction(s);

    }



    if(CONFIG_MPEG_XVMC_DECODER && s->avctx->xvmc_acceleration)

        return ff_xvmc_field_start(s, avctx);



    return 0;

}
