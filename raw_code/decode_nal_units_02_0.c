static int decode_nal_units(H264Context *h, const uint8_t *buf, int buf_size){

    MpegEncContext * const s = &h->s;

    AVCodecContext * const avctx= s->avctx;

    H264Context *hx; ///< thread context

    int buf_index;

    int context_count;

    int next_avc;

    int pass = !(avctx->active_thread_type & FF_THREAD_FRAME);

    int nals_needed=0; ///< number of NALs that need decoding before the next frame thread starts

    int nal_index;



    h->nal_unit_type= 0;



    h->max_contexts = (HAVE_THREADS && (s->avctx->active_thread_type&FF_THREAD_SLICE)) ? avctx->thread_count : 1;

    if(!(s->flags2 & CODEC_FLAG2_CHUNKS)){

        h->current_slice = 0;

        if (!s->first_field)

            s->current_picture_ptr= NULL;

        ff_h264_reset_sei(h);

    }



    for(;pass <= 1;pass++){

        buf_index = 0;

        context_count = 0;

        next_avc = h->is_avc ? 0 : buf_size;

        nal_index = 0;

    for(;;){

        int consumed;

        int dst_length;

        int bit_length;

        uint8_t *ptr;

        int i, nalsize = 0;

        int err;



        if(buf_index >= next_avc) {

            if (buf_index >= buf_size - h->nal_length_size) break;

            nalsize = 0;

            for(i = 0; i < h->nal_length_size; i++)

                nalsize = (nalsize << 8) | buf[buf_index++];

            if(nalsize <= 0 || nalsize > buf_size - buf_index){

                av_log(h->s.avctx, AV_LOG_ERROR, "AVC: nal size %d\n", nalsize);

                break;

            }

            next_avc= buf_index + nalsize;

        } else {

            // start code prefix search

            for(; buf_index + 3 < next_avc; buf_index++){

                // This should always succeed in the first iteration.

                if(buf[buf_index] == 0 && buf[buf_index+1] == 0 && buf[buf_index+2] == 1)

                    break;

            }



            if(buf_index+3 >= buf_size) break;



            buf_index+=3;

            if(buf_index >= next_avc) continue;

        }



        hx = h->thread_context[context_count];



        ptr= ff_h264_decode_nal(hx, buf + buf_index, &dst_length, &consumed, next_avc - buf_index);

        if (ptr==NULL || dst_length < 0){

            return -1;

        }

        i= buf_index + consumed;

        if((s->workaround_bugs & FF_BUG_AUTODETECT) && i+3<next_avc &&

           buf[i]==0x00 && buf[i+1]==0x00 && buf[i+2]==0x01 && buf[i+3]==0xE0)

            s->workaround_bugs |= FF_BUG_TRUNCATED;



        if(!(s->workaround_bugs & FF_BUG_TRUNCATED)){

        while(dst_length > 0 && ptr[dst_length - 1] == 0)

            dst_length--;

        }

        bit_length= !dst_length ? 0 : (8*dst_length - ff_h264_decode_rbsp_trailing(h, ptr + dst_length - 1));



        if(s->avctx->debug&FF_DEBUG_STARTCODE){

            av_log(h->s.avctx, AV_LOG_DEBUG, "NAL %d/%d at %d/%d length %d pass %d\n", hx->nal_unit_type, hx->nal_ref_idc, buf_index, buf_size, dst_length, pass);

        }



        if (h->is_avc && (nalsize != consumed) && nalsize){

            av_log(h->s.avctx, AV_LOG_DEBUG, "AVC: Consumed only %d bytes instead of %d\n", consumed, nalsize);

        }



        buf_index += consumed;

        nal_index++;



        if(pass == 0) {

            // packets can sometimes contain multiple PPS/SPS

            // e.g. two PAFF field pictures in one packet, or a demuxer which splits NALs strangely

            // if so, when frame threading we can't start the next thread until we've read all of them

            switch (hx->nal_unit_type) {

                case NAL_SPS:

                case NAL_PPS:

                    nals_needed = nal_index;

                    break;

                case NAL_IDR_SLICE:

                case NAL_SLICE:

                    init_get_bits(&hx->s.gb, ptr, bit_length);

                    if (!get_ue_golomb(&hx->s.gb))

                        nals_needed = nal_index;

            }

            continue;

        }



        //FIXME do not discard SEI id

        if(avctx->skip_frame >= AVDISCARD_NONREF && h->nal_ref_idc  == 0)

            continue;



      again:

        err = 0;

        switch(hx->nal_unit_type){

        case NAL_IDR_SLICE:

            if (h->nal_unit_type != NAL_IDR_SLICE) {

                av_log(h->s.avctx, AV_LOG_ERROR, "Invalid mix of idr and non-idr slices");

                return -1;

            }

            idr(h); // FIXME ensure we don't lose some frames if there is reordering

        case NAL_SLICE:

            init_get_bits(&hx->s.gb, ptr, bit_length);

            hx->intra_gb_ptr=

            hx->inter_gb_ptr= &hx->s.gb;

            hx->s.data_partitioning = 0;



            if((err = decode_slice_header(hx, h)))

               break;



            if (   h->sei_recovery_frame_cnt >= 0

                && ((h->recovery_frame - h->frame_num) & ((1 << h->sps.log2_max_frame_num)-1)) > h->sei_recovery_frame_cnt) {

                h->recovery_frame = (h->frame_num + h->sei_recovery_frame_cnt) %

                                    (1 << h->sps.log2_max_frame_num);

            }



            s->current_picture_ptr->f.key_frame |=

                    (hx->nal_unit_type == NAL_IDR_SLICE);



            if (h->recovery_frame == h->frame_num) {

                h->sync |= 1;

                h->recovery_frame = -1;

            }



            h->sync |= !!s->current_picture_ptr->f.key_frame;

            h->sync |= 3*!!(s->flags2 & CODEC_FLAG2_SHOW_ALL);

            s->current_picture_ptr->sync = h->sync;



            if (h->current_slice == 1) {

                if(!(s->flags2 & CODEC_FLAG2_CHUNKS)) {

                    decode_postinit(h, nal_index >= nals_needed);

                }



                if (s->avctx->hwaccel && s->avctx->hwaccel->start_frame(s->avctx, NULL, 0) < 0)

                    return -1;

                if(CONFIG_H264_VDPAU_DECODER && s->avctx->codec->capabilities&CODEC_CAP_HWACCEL_VDPAU)

                    ff_vdpau_h264_picture_start(s);

            }



            if(hx->redundant_pic_count==0

               && (avctx->skip_frame < AVDISCARD_NONREF || hx->nal_ref_idc)

               && (avctx->skip_frame < AVDISCARD_BIDIR  || hx->slice_type_nos!=AV_PICTURE_TYPE_B)

               && (avctx->skip_frame < AVDISCARD_NONKEY || hx->slice_type_nos==AV_PICTURE_TYPE_I)

               && avctx->skip_frame < AVDISCARD_ALL){

                if(avctx->hwaccel) {

                    if (avctx->hwaccel->decode_slice(avctx, &buf[buf_index - consumed], consumed) < 0)

                        return -1;

                }else

                if(CONFIG_H264_VDPAU_DECODER && s->avctx->codec->capabilities&CODEC_CAP_HWACCEL_VDPAU){

                    static const uint8_t start_code[] = {0x00, 0x00, 0x01};

                    ff_vdpau_add_data_chunk(s, start_code, sizeof(start_code));

                    ff_vdpau_add_data_chunk(s, &buf[buf_index - consumed], consumed );

                }else

                    context_count++;

            }

            break;

        case NAL_DPA:

            init_get_bits(&hx->s.gb, ptr, bit_length);

            hx->intra_gb_ptr=

            hx->inter_gb_ptr= NULL;



            if ((err = decode_slice_header(hx, h)) < 0)

                break;



            hx->s.data_partitioning = 1;



            break;

        case NAL_DPB:

            init_get_bits(&hx->intra_gb, ptr, bit_length);

            hx->intra_gb_ptr= &hx->intra_gb;

            break;

        case NAL_DPC:

            init_get_bits(&hx->inter_gb, ptr, bit_length);

            hx->inter_gb_ptr= &hx->inter_gb;



            if(hx->redundant_pic_count==0 && hx->intra_gb_ptr && hx->s.data_partitioning

               && s->context_initialized

               && (avctx->skip_frame < AVDISCARD_NONREF || hx->nal_ref_idc)

               && (avctx->skip_frame < AVDISCARD_BIDIR  || hx->slice_type_nos!=AV_PICTURE_TYPE_B)

               && (avctx->skip_frame < AVDISCARD_NONKEY || hx->slice_type_nos==AV_PICTURE_TYPE_I)

               && avctx->skip_frame < AVDISCARD_ALL)

                context_count++;

            break;

        case NAL_SEI:

            init_get_bits(&s->gb, ptr, bit_length);

            ff_h264_decode_sei(h);

            break;

        case NAL_SPS:

            init_get_bits(&s->gb, ptr, bit_length);

            if(ff_h264_decode_seq_parameter_set(h) < 0 && (h->is_avc ? (nalsize != consumed) && nalsize : 1)){

                av_log(h->s.avctx, AV_LOG_DEBUG, "SPS decoding failure, trying alternative mode\n");

                if(h->is_avc) av_assert0(next_avc - buf_index + consumed == nalsize);

                init_get_bits(&s->gb, &buf[buf_index + 1 - consumed], 8*(next_avc - buf_index + consumed));

                ff_h264_decode_seq_parameter_set(h);

            }



            if (s->flags& CODEC_FLAG_LOW_DELAY ||

                (h->sps.bitstream_restriction_flag && !h->sps.num_reorder_frames))

                s->low_delay=1;



            if(avctx->has_b_frames < 2)

                avctx->has_b_frames= !s->low_delay;

            break;

        case NAL_PPS:

            init_get_bits(&s->gb, ptr, bit_length);



            ff_h264_decode_picture_parameter_set(h, bit_length);



            break;

        case NAL_AUD:

        case NAL_END_SEQUENCE:

        case NAL_END_STREAM:

        case NAL_FILLER_DATA:

        case NAL_SPS_EXT:

        case NAL_AUXILIARY_SLICE:

            break;

        default:

            av_log(avctx, AV_LOG_DEBUG, "Unknown NAL code: %d (%d bits)\n", hx->nal_unit_type, bit_length);

        }



        if(context_count == h->max_contexts) {

            execute_decode_slices(h, context_count);

            context_count = 0;

        }



        if (err < 0)

            av_log(h->s.avctx, AV_LOG_ERROR, "decode_slice_header error\n");

        else if(err == 1) {

            /* Slice could not be decoded in parallel mode, copy down

             * NAL unit stuff to context 0 and restart. Note that

             * rbsp_buffer is not transferred, but since we no longer

             * run in parallel mode this should not be an issue. */

            h->nal_unit_type = hx->nal_unit_type;

            h->nal_ref_idc   = hx->nal_ref_idc;

            hx = h;

            goto again;

        }

    }

    }

    if(context_count)

        execute_decode_slices(h, context_count);

    return buf_index;

}
