static int decode_slice_header(H264Context *h){

    MpegEncContext * const s = &h->s;

    int first_mb_in_slice, pps_id;

    int num_ref_idx_active_override_flag;

    static const uint8_t slice_type_map[5]= {P_TYPE, B_TYPE, I_TYPE, SP_TYPE, SI_TYPE};

    int slice_type;

    int default_ref_list_done = 0;



    s->current_picture.reference= h->nal_ref_idc != 0;

    s->dropable= h->nal_ref_idc == 0;



    first_mb_in_slice= get_ue_golomb(&s->gb);



    slice_type= get_ue_golomb(&s->gb);

    if(slice_type > 9){

        av_log(h->s.avctx, AV_LOG_ERROR, "slice type too large (%d) at %d %d\n", h->slice_type, s->mb_x, s->mb_y);

        return -1;

    }

    if(slice_type > 4){

        slice_type -= 5;

        h->slice_type_fixed=1;

    }else

        h->slice_type_fixed=0;



    slice_type= slice_type_map[ slice_type ];

    if (slice_type == I_TYPE

        || (h->slice_num != 0 && slice_type == h->slice_type) ) {

        default_ref_list_done = 1;

    }

    h->slice_type= slice_type;



    s->pict_type= h->slice_type; // to make a few old func happy, it's wrong though



    pps_id= get_ue_golomb(&s->gb);

    if(pps_id>255){

        av_log(h->s.avctx, AV_LOG_ERROR, "pps_id out of range\n");

        return -1;

    }

    h->pps= h->pps_buffer[pps_id];

    if(h->pps.slice_group_count == 0){

        av_log(h->s.avctx, AV_LOG_ERROR, "non existing PPS referenced\n");

        return -1;

    }



    h->sps= h->sps_buffer[ h->pps.sps_id ];

    if(h->sps.log2_max_frame_num == 0){

        av_log(h->s.avctx, AV_LOG_ERROR, "non existing SPS referenced\n");

        return -1;

    }



    if(h->dequant_coeff_pps != pps_id){

        h->dequant_coeff_pps = pps_id;

        init_dequant_tables(h);

    }



    s->mb_width= h->sps.mb_width;

    s->mb_height= h->sps.mb_height * (2 - h->sps.frame_mbs_only_flag);



    h->b_stride=  s->mb_width*4;

    h->b8_stride= s->mb_width*2;



    s->width = 16*s->mb_width - 2*(h->sps.crop_left + h->sps.crop_right );

    if(h->sps.frame_mbs_only_flag)

        s->height= 16*s->mb_height - 2*(h->sps.crop_top  + h->sps.crop_bottom);

    else

        s->height= 16*s->mb_height - 4*(h->sps.crop_top  + h->sps.crop_bottom); //FIXME recheck



    if (s->context_initialized

        && (   s->width != s->avctx->width || s->height != s->avctx->height)) {

        free_tables(h);

        MPV_common_end(s);

    }

    if (!s->context_initialized) {

        if (MPV_common_init(s) < 0)

            return -1;



        if(s->dsp.h264_idct_add == ff_h264_idct_add_c){ //FIXME little ugly

            memcpy(h->zigzag_scan, zigzag_scan, 16*sizeof(uint8_t));

            memcpy(h-> field_scan,  field_scan, 16*sizeof(uint8_t));

        }else{

            int i;

            for(i=0; i<16; i++){

#define T(x) (x>>2) | ((x<<2) & 0xF)

                h->zigzag_scan[i] = T(zigzag_scan[i]);

                h-> field_scan[i] = T( field_scan[i]);

#undef T

            }

        }

        if(s->dsp.h264_idct8_add == ff_h264_idct8_add_c){

            memcpy(h->zigzag_scan8x8,       zigzag_scan8x8,       64*sizeof(uint8_t));

            memcpy(h->zigzag_scan8x8_cavlc, zigzag_scan8x8_cavlc, 64*sizeof(uint8_t));

            memcpy(h->field_scan8x8,        field_scan8x8,        64*sizeof(uint8_t));

            memcpy(h->field_scan8x8_cavlc,  field_scan8x8_cavlc,  64*sizeof(uint8_t));

        }else{

            int i;

            for(i=0; i<64; i++){

#define T(x) (x>>3) | ((x&7)<<3)

                h->zigzag_scan8x8[i]       = T(zigzag_scan8x8[i]);

                h->zigzag_scan8x8_cavlc[i] = T(zigzag_scan8x8_cavlc[i]);

                h->field_scan8x8[i]        = T(field_scan8x8[i]);

                h->field_scan8x8_cavlc[i]  = T(field_scan8x8_cavlc[i]);

#undef T

            }

        }

        if(h->sps.transform_bypass){ //FIXME same ugly

            h->zigzag_scan_q0          = zigzag_scan;

            h->zigzag_scan8x8_q0       = zigzag_scan8x8;

            h->zigzag_scan8x8_cavlc_q0 = zigzag_scan8x8_cavlc;

            h->field_scan_q0           = field_scan;

            h->field_scan8x8_q0        = field_scan8x8;

            h->field_scan8x8_cavlc_q0  = field_scan8x8_cavlc;

        }else{

            h->zigzag_scan_q0          = h->zigzag_scan;

            h->zigzag_scan8x8_q0       = h->zigzag_scan8x8;

            h->zigzag_scan8x8_cavlc_q0 = h->zigzag_scan8x8_cavlc;

            h->field_scan_q0           = h->field_scan;

            h->field_scan8x8_q0        = h->field_scan8x8;

            h->field_scan8x8_cavlc_q0  = h->field_scan8x8_cavlc;

        }



        alloc_tables(h);



        s->avctx->width = s->width;

        s->avctx->height = s->height;

        s->avctx->sample_aspect_ratio= h->sps.sar;

        if(!s->avctx->sample_aspect_ratio.den)

            s->avctx->sample_aspect_ratio.den = 1;



        if(h->sps.timing_info_present_flag){

            s->avctx->time_base= (AVRational){h->sps.num_units_in_tick * 2, h->sps.time_scale};

            if(h->x264_build > 0 && h->x264_build < 44)

                s->avctx->time_base.den *= 2;

            av_reduce(&s->avctx->time_base.num, &s->avctx->time_base.den,

                      s->avctx->time_base.num, s->avctx->time_base.den, 1<<30);

        }

    }



    if(h->slice_num == 0){

        if(frame_start(h) < 0)

            return -1;

    }



    s->current_picture_ptr->frame_num= //FIXME frame_num cleanup

    h->frame_num= get_bits(&s->gb, h->sps.log2_max_frame_num);



    h->mb_mbaff = 0;

    h->mb_aff_frame = 0;

    if(h->sps.frame_mbs_only_flag){

        s->picture_structure= PICT_FRAME;

    }else{

        if(get_bits1(&s->gb)) { //field_pic_flag

            s->picture_structure= PICT_TOP_FIELD + get_bits1(&s->gb); //bottom_field_flag

            av_log(h->s.avctx, AV_LOG_ERROR, "PAFF interlacing is not implemented\n");

        } else {

            s->picture_structure= PICT_FRAME;

            h->mb_aff_frame = h->sps.mb_aff;

        }

    }



    s->resync_mb_x = s->mb_x = first_mb_in_slice % s->mb_width;

    s->resync_mb_y = s->mb_y = (first_mb_in_slice / s->mb_width) << h->mb_aff_frame;

    if(s->mb_y >= s->mb_height){

        return -1;

    }



    if(s->picture_structure==PICT_FRAME){

        h->curr_pic_num=   h->frame_num;

        h->max_pic_num= 1<< h->sps.log2_max_frame_num;

    }else{

        h->curr_pic_num= 2*h->frame_num;

        h->max_pic_num= 1<<(h->sps.log2_max_frame_num + 1);

    }



    if(h->nal_unit_type == NAL_IDR_SLICE){

        get_ue_golomb(&s->gb); /* idr_pic_id */

    }



    if(h->sps.poc_type==0){

        h->poc_lsb= get_bits(&s->gb, h->sps.log2_max_poc_lsb);



        if(h->pps.pic_order_present==1 && s->picture_structure==PICT_FRAME){

            h->delta_poc_bottom= get_se_golomb(&s->gb);

        }

    }



    if(h->sps.poc_type==1 && !h->sps.delta_pic_order_always_zero_flag){

        h->delta_poc[0]= get_se_golomb(&s->gb);



        if(h->pps.pic_order_present==1 && s->picture_structure==PICT_FRAME)

            h->delta_poc[1]= get_se_golomb(&s->gb);

    }



    init_poc(h);



    if(h->pps.redundant_pic_cnt_present){

        h->redundant_pic_count= get_ue_golomb(&s->gb);

    }



    //set defaults, might be overriden a few line later

    h->ref_count[0]= h->pps.ref_count[0];

    h->ref_count[1]= h->pps.ref_count[1];



    if(h->slice_type == P_TYPE || h->slice_type == SP_TYPE || h->slice_type == B_TYPE){

        if(h->slice_type == B_TYPE){

            h->direct_spatial_mv_pred= get_bits1(&s->gb);

            if(h->sps.mb_aff && h->direct_spatial_mv_pred)

                av_log(h->s.avctx, AV_LOG_ERROR, "MBAFF + spatial direct mode is not implemented\n");

        }

        num_ref_idx_active_override_flag= get_bits1(&s->gb);



        if(num_ref_idx_active_override_flag){

            h->ref_count[0]= get_ue_golomb(&s->gb) + 1;

            if(h->slice_type==B_TYPE)

                h->ref_count[1]= get_ue_golomb(&s->gb) + 1;



            if(h->ref_count[0] > 32 || h->ref_count[1] > 32){

                av_log(h->s.avctx, AV_LOG_ERROR, "reference overflow\n");

                return -1;

            }

        }

    }



    if(!default_ref_list_done){

        fill_default_ref_list(h);

    }



    if(decode_ref_pic_list_reordering(h) < 0)

        return -1;



    if(   (h->pps.weighted_pred          && (h->slice_type == P_TYPE || h->slice_type == SP_TYPE ))

       || (h->pps.weighted_bipred_idc==1 && h->slice_type==B_TYPE ) )

        pred_weight_table(h);

    else if(h->pps.weighted_bipred_idc==2 && h->slice_type==B_TYPE)

        implicit_weight_table(h);

    else

        h->use_weight = 0;



    if(s->current_picture.reference)

        decode_ref_pic_marking(h);



    if(FRAME_MBAFF)

        fill_mbaff_ref_list(h);



    if( h->slice_type != I_TYPE && h->slice_type != SI_TYPE && h->pps.cabac )

        h->cabac_init_idc = get_ue_golomb(&s->gb);



    h->last_qscale_diff = 0;

    s->qscale = h->pps.init_qp + get_se_golomb(&s->gb);

    if(s->qscale<0 || s->qscale>51){

        av_log(s->avctx, AV_LOG_ERROR, "QP %d out of range\n", s->qscale);

        return -1;

    }

    h->chroma_qp = get_chroma_qp(h->pps.chroma_qp_index_offset, s->qscale);

    //FIXME qscale / qp ... stuff

    if(h->slice_type == SP_TYPE){

        get_bits1(&s->gb); /* sp_for_switch_flag */

    }

    if(h->slice_type==SP_TYPE || h->slice_type == SI_TYPE){

        get_se_golomb(&s->gb); /* slice_qs_delta */

    }



    h->deblocking_filter = 1;

    h->slice_alpha_c0_offset = 0;

    h->slice_beta_offset = 0;

    if( h->pps.deblocking_filter_parameters_present ) {

        h->deblocking_filter= get_ue_golomb(&s->gb);

        if(h->deblocking_filter < 2)

            h->deblocking_filter^= 1; // 1<->0



        if( h->deblocking_filter ) {

            h->slice_alpha_c0_offset = get_se_golomb(&s->gb) << 1;

            h->slice_beta_offset = get_se_golomb(&s->gb) << 1;

        }

    }

    if(   s->avctx->skip_loop_filter >= AVDISCARD_ALL

       ||(s->avctx->skip_loop_filter >= AVDISCARD_NONKEY && h->slice_type != I_TYPE)

       ||(s->avctx->skip_loop_filter >= AVDISCARD_BIDIR  && h->slice_type == B_TYPE)

       ||(s->avctx->skip_loop_filter >= AVDISCARD_NONREF && h->nal_ref_idc == 0))

        h->deblocking_filter= 0;



#if 0 //FMO

    if( h->pps.num_slice_groups > 1  && h->pps.mb_slice_group_map_type >= 3 && h->pps.mb_slice_group_map_type <= 5)

        slice_group_change_cycle= get_bits(&s->gb, ?);

#endif



    h->slice_num++;



    h->emu_edge_width= (s->flags&CODEC_FLAG_EMU_EDGE) ? 0 : 16;

    h->emu_edge_height= FRAME_MBAFF ? 0 : h->emu_edge_width;



    if(s->avctx->debug&FF_DEBUG_PICT_INFO){

        av_log(h->s.avctx, AV_LOG_DEBUG, "slice:%d %s mb:%d %c pps:%d frame:%d poc:%d/%d ref:%d/%d qp:%d loop:%d:%d:%d weight:%d%s\n",

               h->slice_num,

               (s->picture_structure==PICT_FRAME ? "F" : s->picture_structure==PICT_TOP_FIELD ? "T" : "B"),

               first_mb_in_slice,

               av_get_pict_type_char(h->slice_type),

               pps_id, h->frame_num,

               s->current_picture_ptr->field_poc[0], s->current_picture_ptr->field_poc[1],

               h->ref_count[0], h->ref_count[1],

               s->qscale,

               h->deblocking_filter, h->slice_alpha_c0_offset/2, h->slice_beta_offset/2,

               h->use_weight,

               h->use_weight==1 && h->use_weight_chroma ? "c" : ""

               );

    }



    if((s->avctx->flags2 & CODEC_FLAG2_FAST) && !s->current_picture.reference){

        s->me.qpel_put= s->dsp.put_2tap_qpel_pixels_tab;

        s->me.qpel_avg= s->dsp.avg_2tap_qpel_pixels_tab;

    }else{

        s->me.qpel_put= s->dsp.put_h264_qpel_pixels_tab;

        s->me.qpel_avg= s->dsp.avg_h264_qpel_pixels_tab;

    }



    return 0;

}
