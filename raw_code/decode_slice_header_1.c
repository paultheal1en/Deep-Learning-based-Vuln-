static int decode_slice_header(H264Context *h, H264Context *h0){

    MpegEncContext * const s = &h->s;

    MpegEncContext * const s0 = &h0->s;

    unsigned int first_mb_in_slice;

    unsigned int pps_id;

    int num_ref_idx_active_override_flag;

    unsigned int slice_type, tmp, i, j;

    int default_ref_list_done = 0;

    int last_pic_structure;



    s->dropable= h->nal_ref_idc == 0;



    if((s->avctx->flags2 & CODEC_FLAG2_FAST) && !h->nal_ref_idc){

        s->me.qpel_put= s->dsp.put_2tap_qpel_pixels_tab;

        s->me.qpel_avg= s->dsp.avg_2tap_qpel_pixels_tab;

    }else{

        s->me.qpel_put= s->dsp.put_h264_qpel_pixels_tab;

        s->me.qpel_avg= s->dsp.avg_h264_qpel_pixels_tab;

    }



    first_mb_in_slice= get_ue_golomb(&s->gb);



    if((s->flags2 & CODEC_FLAG2_CHUNKS) && first_mb_in_slice == 0){

        h0->current_slice = 0;

        if (!s0->first_field)

            s->current_picture_ptr= NULL;

    }



    slice_type= get_ue_golomb_31(&s->gb);

    if(slice_type > 9){

        av_log(h->s.avctx, AV_LOG_ERROR, "slice type too large (%d) at %d %d\n", h->slice_type, s->mb_x, s->mb_y);

        return -1;

    }

    if(slice_type > 4){

        slice_type -= 5;

        h->slice_type_fixed=1;

    }else

        h->slice_type_fixed=0;



    slice_type= golomb_to_pict_type[ slice_type ];

    if (slice_type == FF_I_TYPE

        || (h0->current_slice != 0 && slice_type == h0->last_slice_type) ) {

        default_ref_list_done = 1;

    }

    h->slice_type= slice_type;

    h->slice_type_nos= slice_type & 3;



    s->pict_type= h->slice_type; // to make a few old functions happy, it's wrong though

    if (s->pict_type == FF_B_TYPE && s0->last_picture_ptr == NULL) {

        av_log(h->s.avctx, AV_LOG_ERROR,

               "B picture before any references, skipping\n");

        return -1;

    }



    pps_id= get_ue_golomb(&s->gb);

    if(pps_id>=MAX_PPS_COUNT){

        av_log(h->s.avctx, AV_LOG_ERROR, "pps_id out of range\n");

        return -1;

    }

    if(!h0->pps_buffers[pps_id]) {

        av_log(h->s.avctx, AV_LOG_ERROR, "non-existing PPS referenced\n");

        return -1;

    }

    h->pps= *h0->pps_buffers[pps_id];



    if(!h0->sps_buffers[h->pps.sps_id]) {

        av_log(h->s.avctx, AV_LOG_ERROR, "non-existing SPS referenced\n");

        return -1;

    }

    h->sps = *h0->sps_buffers[h->pps.sps_id];



    if(h == h0 && h->dequant_coeff_pps != pps_id){

        h->dequant_coeff_pps = pps_id;

        init_dequant_tables(h);

    }



    s->mb_width= h->sps.mb_width;

    s->mb_height= h->sps.mb_height * (2 - h->sps.frame_mbs_only_flag);



    h->b_stride=  s->mb_width*4;

    h->b8_stride= s->mb_width*2;



    s->width = 16*s->mb_width - 2*FFMIN(h->sps.crop_right, 7);

    if(h->sps.frame_mbs_only_flag)

        s->height= 16*s->mb_height - 2*FFMIN(h->sps.crop_bottom, 7);

    else

        s->height= 16*s->mb_height - 4*FFMIN(h->sps.crop_bottom, 3);



    if (s->context_initialized

        && (   s->width != s->avctx->width || s->height != s->avctx->height)) {

        if(h != h0)

            return -1;   // width / height changed during parallelized decoding

        free_tables(h);

        flush_dpb(s->avctx);

        MPV_common_end(s);

    }

    if (!s->context_initialized) {

        if(h != h0)

            return -1;  // we cant (re-)initialize context during parallel decoding

        if (MPV_common_init(s) < 0)

            return -1;

        s->first_field = 0;



        init_scan_tables(h);

        alloc_tables(h);



        for(i = 1; i < s->avctx->thread_count; i++) {

            H264Context *c;

            c = h->thread_context[i] = av_malloc(sizeof(H264Context));

            memcpy(c, h->s.thread_context[i], sizeof(MpegEncContext));

            memset(&c->s + 1, 0, sizeof(H264Context) - sizeof(MpegEncContext));

            c->sps = h->sps;

            c->pps = h->pps;

            init_scan_tables(c);

            clone_tables(c, h);

        }



        for(i = 0; i < s->avctx->thread_count; i++)

            if(context_init(h->thread_context[i]) < 0)

                return -1;



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



    h->frame_num= get_bits(&s->gb, h->sps.log2_max_frame_num);



    h->mb_mbaff = 0;

    h->mb_aff_frame = 0;

    last_pic_structure = s0->picture_structure;

    if(h->sps.frame_mbs_only_flag){

        s->picture_structure= PICT_FRAME;

    }else{

        if(get_bits1(&s->gb)) { //field_pic_flag

            s->picture_structure= PICT_TOP_FIELD + get_bits1(&s->gb); //bottom_field_flag

        } else {

            s->picture_structure= PICT_FRAME;

            h->mb_aff_frame = h->sps.mb_aff;

        }

    }

    h->mb_field_decoding_flag= s->picture_structure != PICT_FRAME;



    if(h0->current_slice == 0){

        while(h->frame_num !=  h->prev_frame_num &&

              h->frame_num != (h->prev_frame_num+1)%(1<<h->sps.log2_max_frame_num)){

            av_log(NULL, AV_LOG_DEBUG, "Frame num gap %d %d\n", h->frame_num, h->prev_frame_num);

            frame_start(h);

            h->prev_frame_num++;

            h->prev_frame_num %= 1<<h->sps.log2_max_frame_num;

            s->current_picture_ptr->frame_num= h->prev_frame_num;

            execute_ref_pic_marking(h, NULL, 0);

        }



        /* See if we have a decoded first field looking for a pair... */

        if (s0->first_field) {

            assert(s0->current_picture_ptr);

            assert(s0->current_picture_ptr->data[0]);

            assert(s0->current_picture_ptr->reference != DELAYED_PIC_REF);



            /* figure out if we have a complementary field pair */

            if (!FIELD_PICTURE || s->picture_structure == last_pic_structure) {

                /*

                 * Previous field is unmatched. Don't display it, but let it

                 * remain for reference if marked as such.

                 */

                s0->current_picture_ptr = NULL;

                s0->first_field = FIELD_PICTURE;



            } else {

                if (h->nal_ref_idc &&

                        s0->current_picture_ptr->reference &&

                        s0->current_picture_ptr->frame_num != h->frame_num) {

                    /*

                     * This and previous field were reference, but had

                     * different frame_nums. Consider this field first in

                     * pair. Throw away previous field except for reference

                     * purposes.

                     */

                    s0->first_field = 1;

                    s0->current_picture_ptr = NULL;



                } else {

                    /* Second field in complementary pair */

                    s0->first_field = 0;

                }

            }



        } else {

            /* Frame or first field in a potentially complementary pair */

            assert(!s0->current_picture_ptr);

            s0->first_field = FIELD_PICTURE;

        }



        if((!FIELD_PICTURE || s0->first_field) && frame_start(h) < 0) {

            s0->first_field = 0;

            return -1;

        }

    }

    if(h != h0)

        clone_slice(h, h0);



    s->current_picture_ptr->frame_num= h->frame_num; //FIXME frame_num cleanup



    assert(s->mb_num == s->mb_width * s->mb_height);

    if(first_mb_in_slice << FIELD_OR_MBAFF_PICTURE >= s->mb_num ||

       first_mb_in_slice                    >= s->mb_num){

        av_log(h->s.avctx, AV_LOG_ERROR, "first_mb_in_slice overflow\n");

        return -1;

    }

    s->resync_mb_x = s->mb_x = first_mb_in_slice % s->mb_width;

    s->resync_mb_y = s->mb_y = (first_mb_in_slice / s->mb_width) << FIELD_OR_MBAFF_PICTURE;

    if (s->picture_structure == PICT_BOTTOM_FIELD)

        s->resync_mb_y = s->mb_y = s->mb_y + 1;

    assert(s->mb_y < s->mb_height);



    if(s->picture_structure==PICT_FRAME){

        h->curr_pic_num=   h->frame_num;

        h->max_pic_num= 1<< h->sps.log2_max_frame_num;

    }else{

        h->curr_pic_num= 2*h->frame_num + 1;

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



    //set defaults, might be overridden a few lines later

    h->ref_count[0]= h->pps.ref_count[0];

    h->ref_count[1]= h->pps.ref_count[1];



    if(h->slice_type_nos != FF_I_TYPE){

        if(h->slice_type_nos == FF_B_TYPE){

            h->direct_spatial_mv_pred= get_bits1(&s->gb);

        }

        num_ref_idx_active_override_flag= get_bits1(&s->gb);



        if(num_ref_idx_active_override_flag){

            h->ref_count[0]= get_ue_golomb(&s->gb) + 1;

            if(h->slice_type_nos==FF_B_TYPE)

                h->ref_count[1]= get_ue_golomb(&s->gb) + 1;



            if(h->ref_count[0]-1 > 32-1 || h->ref_count[1]-1 > 32-1){

                av_log(h->s.avctx, AV_LOG_ERROR, "reference overflow\n");

                h->ref_count[0]= h->ref_count[1]= 1;

                return -1;

            }

        }

        if(h->slice_type_nos == FF_B_TYPE)

            h->list_count= 2;

        else

            h->list_count= 1;

    }else

        h->list_count= 0;



    if(!default_ref_list_done){

        fill_default_ref_list(h);

    }



    if(h->slice_type_nos!=FF_I_TYPE && decode_ref_pic_list_reordering(h) < 0)

        return -1;



    if(h->slice_type_nos!=FF_I_TYPE){

        s->last_picture_ptr= &h->ref_list[0][0];

        ff_copy_picture(&s->last_picture, s->last_picture_ptr);

    }

    if(h->slice_type_nos==FF_B_TYPE){

        s->next_picture_ptr= &h->ref_list[1][0];

        ff_copy_picture(&s->next_picture, s->next_picture_ptr);

    }



    if(   (h->pps.weighted_pred          && h->slice_type_nos == FF_P_TYPE )

       ||  (h->pps.weighted_bipred_idc==1 && h->slice_type_nos== FF_B_TYPE ) )

        pred_weight_table(h);

    else if(h->pps.weighted_bipred_idc==2 && h->slice_type_nos== FF_B_TYPE)

        implicit_weight_table(h);

    else {

        h->use_weight = 0;

        for (i = 0; i < 2; i++) {

            h->luma_weight_flag[i]   = 0;

            h->chroma_weight_flag[i] = 0;

        }

    }



    if(h->nal_ref_idc)

        decode_ref_pic_marking(h0, &s->gb);



    if(FRAME_MBAFF)

        fill_mbaff_ref_list(h);



    if(h->slice_type_nos==FF_B_TYPE && !h->direct_spatial_mv_pred)

        direct_dist_scale_factor(h);

    direct_ref_list_init(h);



    if( h->slice_type_nos != FF_I_TYPE && h->pps.cabac ){

        tmp = get_ue_golomb_31(&s->gb);

        if(tmp > 2){

            av_log(s->avctx, AV_LOG_ERROR, "cabac_init_idc overflow\n");

            return -1;

        }

        h->cabac_init_idc= tmp;

    }



    h->last_qscale_diff = 0;

    tmp = h->pps.init_qp + get_se_golomb(&s->gb);

    if(tmp>51){

        av_log(s->avctx, AV_LOG_ERROR, "QP %u out of range\n", tmp);

        return -1;

    }

    s->qscale= tmp;

    h->chroma_qp[0] = get_chroma_qp(h, 0, s->qscale);

    h->chroma_qp[1] = get_chroma_qp(h, 1, s->qscale);

    //FIXME qscale / qp ... stuff

    if(h->slice_type == FF_SP_TYPE){

        get_bits1(&s->gb); /* sp_for_switch_flag */

    }

    if(h->slice_type==FF_SP_TYPE || h->slice_type == FF_SI_TYPE){

        get_se_golomb(&s->gb); /* slice_qs_delta */

    }



    h->deblocking_filter = 1;

    h->slice_alpha_c0_offset = 0;

    h->slice_beta_offset = 0;

    if( h->pps.deblocking_filter_parameters_present ) {

        tmp= get_ue_golomb_31(&s->gb);

        if(tmp > 2){

            av_log(s->avctx, AV_LOG_ERROR, "deblocking_filter_idc %u out of range\n", tmp);

            return -1;

        }

        h->deblocking_filter= tmp;

        if(h->deblocking_filter < 2)

            h->deblocking_filter^= 1; // 1<->0



        if( h->deblocking_filter ) {

            h->slice_alpha_c0_offset = get_se_golomb(&s->gb) << 1;

            h->slice_beta_offset = get_se_golomb(&s->gb) << 1;

        }

    }



    if(   s->avctx->skip_loop_filter >= AVDISCARD_ALL

       ||(s->avctx->skip_loop_filter >= AVDISCARD_NONKEY && h->slice_type_nos != FF_I_TYPE)

       ||(s->avctx->skip_loop_filter >= AVDISCARD_BIDIR  && h->slice_type_nos == FF_B_TYPE)

       ||(s->avctx->skip_loop_filter >= AVDISCARD_NONREF && h->nal_ref_idc == 0))

        h->deblocking_filter= 0;



    if(h->deblocking_filter == 1 && h0->max_contexts > 1) {

        if(s->avctx->flags2 & CODEC_FLAG2_FAST) {

            /* Cheat slightly for speed:

               Do not bother to deblock across slices. */

            h->deblocking_filter = 2;

        } else {

            h0->max_contexts = 1;

            if(!h0->single_decode_warning) {

                av_log(s->avctx, AV_LOG_INFO, "Cannot parallelize deblocking type 1, decoding such frames in sequential order\n");

                h0->single_decode_warning = 1;

            }

            if(h != h0)

                return 1; // deblocking switched inside frame

        }

    }



#if 0 //FMO

    if( h->pps.num_slice_groups > 1  && h->pps.mb_slice_group_map_type >= 3 && h->pps.mb_slice_group_map_type <= 5)

        slice_group_change_cycle= get_bits(&s->gb, ?);

#endif



    h0->last_slice_type = slice_type;

    h->slice_num = ++h0->current_slice;

    if(h->slice_num >= MAX_SLICES){

        av_log(s->avctx, AV_LOG_ERROR, "Too many slices, increase MAX_SLICES and recompile\n");

    }



    for(j=0; j<2; j++){

        int *ref2frm= h->ref2frm[h->slice_num&(MAX_SLICES-1)][j];

        ref2frm[0]=

        ref2frm[1]= -1;

        for(i=0; i<16; i++)

            ref2frm[i+2]= 4*h->ref_list[j][i].frame_num

                          +(h->ref_list[j][i].reference&3);

        ref2frm[18+0]=

        ref2frm[18+1]= -1;

        for(i=16; i<48; i++)

            ref2frm[i+4]= 4*h->ref_list[j][i].frame_num

                          +(h->ref_list[j][i].reference&3);

    }



    h->emu_edge_width= (s->flags&CODEC_FLAG_EMU_EDGE) ? 0 : 16;

    h->emu_edge_height= (FRAME_MBAFF || FIELD_PICTURE) ? 0 : h->emu_edge_width;



    s->avctx->refs= h->sps.ref_frame_count;



    if(s->avctx->debug&FF_DEBUG_PICT_INFO){

        av_log(h->s.avctx, AV_LOG_DEBUG, "slice:%d %s mb:%d %c%s%s pps:%u frame:%d poc:%d/%d ref:%d/%d qp:%d loop:%d:%d:%d weight:%d%s %s\n",

               h->slice_num,

               (s->picture_structure==PICT_FRAME ? "F" : s->picture_structure==PICT_TOP_FIELD ? "T" : "B"),

               first_mb_in_slice,

               av_get_pict_type_char(h->slice_type), h->slice_type_fixed ? " fix" : "", h->nal_unit_type == NAL_IDR_SLICE ? " IDR" : "",

               pps_id, h->frame_num,

               s->current_picture_ptr->field_poc[0], s->current_picture_ptr->field_poc[1],

               h->ref_count[0], h->ref_count[1],

               s->qscale,

               h->deblocking_filter, h->slice_alpha_c0_offset/2, h->slice_beta_offset/2,

               h->use_weight,

               h->use_weight==1 && h->use_weight_chroma ? "c" : "",

               h->slice_type == FF_B_TYPE ? (h->direct_spatial_mv_pred ? "SPAT" : "TEMP") : ""

               );

    }



    return 0;

}
