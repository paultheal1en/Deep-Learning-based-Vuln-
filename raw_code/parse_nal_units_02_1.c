static inline int parse_nal_units(AVCodecParserContext *s,

                                  AVCodecContext *avctx,

                                  const uint8_t *buf, int buf_size)

{

    H264Context *h = s->priv_data;

    const uint8_t *buf_end = buf + buf_size;

    unsigned int pps_id;

    unsigned int slice_type;

    int state;

    const uint8_t *ptr;



    /* set some sane default values */

    s->pict_type = FF_I_TYPE;

    s->key_frame = 0;



    h->s.avctx= avctx;

    h->sei_recovery_frame_cnt = -1;

    h->sei_dpb_output_delay         =  0;

    h->sei_cpb_removal_delay        = -1;

    h->sei_buffering_period_present =  0;



    for(;;) {

        int src_length, dst_length, consumed;

        buf = ff_find_start_code(buf, buf_end, &state);

        if(buf >= buf_end)

            break;

        --buf;

        src_length = buf_end - buf;

        switch (state & 0x1f) {

        case NAL_SLICE:

        case NAL_IDR_SLICE:

            // Do not walk the whole buffer just to decode slice header

            if (src_length > 20)

                src_length = 20;

            break;

        }

        ptr= ff_h264_decode_nal(h, buf, &dst_length, &consumed, src_length);

        if (ptr==NULL || dst_length < 0)

            break;



        init_get_bits(&h->s.gb, ptr, 8*dst_length);

        switch(h->nal_unit_type) {

        case NAL_SPS:

            ff_h264_decode_seq_parameter_set(h);

            break;

        case NAL_PPS:

            ff_h264_decode_picture_parameter_set(h, h->s.gb.size_in_bits);

            break;

        case NAL_SEI:

            ff_h264_decode_sei(h);

            break;

        case NAL_IDR_SLICE:

            s->key_frame = 1;

            /* fall through */

        case NAL_SLICE:

            get_ue_golomb(&h->s.gb);  // skip first_mb_in_slice

            slice_type = get_ue_golomb_31(&h->s.gb);

            s->pict_type = golomb_to_pict_type[slice_type % 5];

            if (h->sei_recovery_frame_cnt >= 0) {

                /* key frame, since recovery_frame_cnt is set */

                s->key_frame = 1;

            }

            pps_id= get_ue_golomb(&h->s.gb);

            if(pps_id>=MAX_PPS_COUNT) {

                av_log(h->s.avctx, AV_LOG_ERROR, "pps_id out of range\n");

                return -1;

            }

            if(!h->pps_buffers[pps_id]) {

                av_log(h->s.avctx, AV_LOG_ERROR, "non-existing PPS referenced\n");

                return -1;

            }

            h->pps= *h->pps_buffers[pps_id];

            if(!h->sps_buffers[h->pps.sps_id]) {

                av_log(h->s.avctx, AV_LOG_ERROR, "non-existing SPS referenced\n");

                return -1;

            }

            h->sps = *h->sps_buffers[h->pps.sps_id];

            h->frame_num = get_bits(&h->s.gb, h->sps.log2_max_frame_num);



            if(h->sps.frame_mbs_only_flag){

                h->s.picture_structure= PICT_FRAME;

            }else{

                if(get_bits1(&h->s.gb)) { //field_pic_flag

                    h->s.picture_structure= PICT_TOP_FIELD + get_bits1(&h->s.gb); //bottom_field_flag

                } else {

                    h->s.picture_structure= PICT_FRAME;

                }

            }



            if(h->sps.pic_struct_present_flag) {

                switch (h->sei_pic_struct) {

                    case SEI_PIC_STRUCT_TOP_FIELD:

                    case SEI_PIC_STRUCT_BOTTOM_FIELD:

                        s->repeat_pict = 0;

                        break;

                    case SEI_PIC_STRUCT_FRAME:

                    case SEI_PIC_STRUCT_TOP_BOTTOM:

                    case SEI_PIC_STRUCT_BOTTOM_TOP:

                        s->repeat_pict = 1;

                        break;

                    case SEI_PIC_STRUCT_TOP_BOTTOM_TOP:

                    case SEI_PIC_STRUCT_BOTTOM_TOP_BOTTOM:

                        s->repeat_pict = 2;

                        break;

                    case SEI_PIC_STRUCT_FRAME_DOUBLING:

                        s->repeat_pict = 3;

                        break;

                    case SEI_PIC_STRUCT_FRAME_TRIPLING:

                        s->repeat_pict = 5;

                        break;

                    default:

                        s->repeat_pict = h->s.picture_structure == PICT_FRAME ? 1 : 0;

                        break;

                }

            } else {

                s->repeat_pict = h->s.picture_structure == PICT_FRAME ? 1 : 0;

            }



            return 0; /* no need to evaluate the rest */

        }

        buf += consumed;

    }

    /* didn't find a picture! */

    av_log(h->s.avctx, AV_LOG_ERROR, "missing picture in access unit\n");

    return -1;

}
