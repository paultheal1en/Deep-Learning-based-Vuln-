static int decode_nal_units(H264Context *h, uint8_t *buf, int buf_size){

    MpegEncContext * const s = &h->s;

    AVCodecContext * const avctx= s->avctx;

    int buf_index=0;

#if 0

    int i;

    for(i=0; i<50; i++){

        av_log(NULL, AV_LOG_ERROR,"%02X ", buf[i]);

    }

#endif

    h->slice_num = 0;

    s->current_picture_ptr= NULL;

    for(;;){

        int consumed;

        int dst_length;

        int bit_length;

        uint8_t *ptr;

        int i, nalsize = 0;



      if(h->is_avc) {

        if(buf_index >= buf_size) break;

        nalsize = 0;

        for(i = 0; i < h->nal_length_size; i++)

            nalsize = (nalsize << 8) | buf[buf_index++];

        if(nalsize <= 1){

            if(nalsize == 1){

                buf_index++;

                continue;

            }else{

                av_log(h->s.avctx, AV_LOG_ERROR, "AVC: nal size %d\n", nalsize);

                break;

            }

        }

      } else {

        // start code prefix search

        for(; buf_index + 3 < buf_size; buf_index++){

            // this should allways succeed in the first iteration

            if(buf[buf_index] == 0 && buf[buf_index+1] == 0 && buf[buf_index+2] == 1)

                break;

        }



        if(buf_index+3 >= buf_size) break;



        buf_index+=3;

      }



        ptr= decode_nal(h, buf + buf_index, &dst_length, &consumed, h->is_avc ? nalsize : buf_size - buf_index);

        while(ptr[dst_length - 1] == 0 && dst_length > 1)

            dst_length--;

        bit_length= 8*dst_length - decode_rbsp_trailing(ptr + dst_length - 1);



        if(s->avctx->debug&FF_DEBUG_STARTCODE){

            av_log(h->s.avctx, AV_LOG_DEBUG, "NAL %d at %d/%d length %d\n", h->nal_unit_type, buf_index, buf_size, dst_length);

        }



        if (h->is_avc && (nalsize != consumed))

            av_log(h->s.avctx, AV_LOG_ERROR, "AVC: Consumed only %d bytes instead of %d\n", consumed, nalsize);



        buf_index += consumed;



        if(  (s->hurry_up == 1 && h->nal_ref_idc  == 0) //FIXME dont discard SEI id

           ||(avctx->skip_frame >= AVDISCARD_NONREF && h->nal_ref_idc  == 0))

            continue;



        switch(h->nal_unit_type){

        case NAL_IDR_SLICE:

            idr(h); //FIXME ensure we don't loose some frames if there is reordering

        case NAL_SLICE:

            init_get_bits(&s->gb, ptr, bit_length);

            h->intra_gb_ptr=

            h->inter_gb_ptr= &s->gb;

            s->data_partitioning = 0;



            if(decode_slice_header(h) < 0){

                av_log(h->s.avctx, AV_LOG_ERROR, "decode_slice_header error\n");

                break;

            }

            s->current_picture_ptr->key_frame= (h->nal_unit_type == NAL_IDR_SLICE);

            if(h->redundant_pic_count==0 && s->hurry_up < 5

               && (avctx->skip_frame < AVDISCARD_NONREF || h->nal_ref_idc)

               && (avctx->skip_frame < AVDISCARD_BIDIR  || h->slice_type!=B_TYPE)

               && (avctx->skip_frame < AVDISCARD_NONKEY || h->slice_type==I_TYPE)

               && avctx->skip_frame < AVDISCARD_ALL)

                decode_slice(h);

            break;

        case NAL_DPA:

            init_get_bits(&s->gb, ptr, bit_length);

            h->intra_gb_ptr=

            h->inter_gb_ptr= NULL;

            s->data_partitioning = 1;



            if(decode_slice_header(h) < 0){

                av_log(h->s.avctx, AV_LOG_ERROR, "decode_slice_header error\n");

            }

            break;

        case NAL_DPB:

            init_get_bits(&h->intra_gb, ptr, bit_length);

            h->intra_gb_ptr= &h->intra_gb;

            break;

        case NAL_DPC:

            init_get_bits(&h->inter_gb, ptr, bit_length);

            h->inter_gb_ptr= &h->inter_gb;



            if(h->redundant_pic_count==0 && h->intra_gb_ptr && s->data_partitioning

               && s->context_initialized

               && s->hurry_up < 5

               && (avctx->skip_frame < AVDISCARD_NONREF || h->nal_ref_idc)

               && (avctx->skip_frame < AVDISCARD_BIDIR  || h->slice_type!=B_TYPE)

               && (avctx->skip_frame < AVDISCARD_NONKEY || h->slice_type==I_TYPE)

               && avctx->skip_frame < AVDISCARD_ALL)

                decode_slice(h);

            break;

        case NAL_SEI:

            init_get_bits(&s->gb, ptr, bit_length);

            decode_sei(h);

            break;

        case NAL_SPS:

            init_get_bits(&s->gb, ptr, bit_length);

            decode_seq_parameter_set(h);



            if(s->flags& CODEC_FLAG_LOW_DELAY)

                s->low_delay=1;



            if(avctx->has_b_frames < 2)

                avctx->has_b_frames= !s->low_delay;

            break;

        case NAL_PPS:

            init_get_bits(&s->gb, ptr, bit_length);



            decode_picture_parameter_set(h, bit_length);



            break;

        case NAL_AUD:

        case NAL_END_SEQUENCE:

        case NAL_END_STREAM:

        case NAL_FILLER_DATA:

        case NAL_SPS_EXT:

        case NAL_AUXILIARY_SLICE:

            break;

        default:

            av_log(avctx, AV_LOG_ERROR, "Unknown NAL code: %d\n", h->nal_unit_type);

        }

    }



    if(!s->current_picture_ptr) return buf_index; //no frame



    s->current_picture_ptr->qscale_type= FF_QSCALE_TYPE_H264;

    s->current_picture_ptr->pict_type= s->pict_type;



    h->prev_frame_num_offset= h->frame_num_offset;

    h->prev_frame_num= h->frame_num;

    if(s->current_picture_ptr->reference){

        h->prev_poc_msb= h->poc_msb;

        h->prev_poc_lsb= h->poc_lsb;

    }

    if(s->current_picture_ptr->reference)

        execute_ref_pic_marking(h, h->mmco, h->mmco_index);



    ff_er_frame_end(s);



    MPV_frame_end(s);



    return buf_index;

}
