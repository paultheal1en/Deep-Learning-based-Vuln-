static int vaapi_encode_h265_init_sequence_params(AVCodecContext *avctx)

{

    VAAPIEncodeContext                 *ctx = avctx->priv_data;

    VAEncSequenceParameterBufferHEVC  *vseq = ctx->codec_sequence_params;

    VAEncPictureParameterBufferHEVC   *vpic = ctx->codec_picture_params;

    VAAPIEncodeH265Context            *priv = ctx->priv_data;

    VAAPIEncodeH265MiscSequenceParams *mseq = &priv->misc_sequence_params;

    int i;



    {

        // general_profile_space == 0.

        vseq->general_profile_idc = 1; // Main profile (ctx->codec_profile?)

        vseq->general_tier_flag = 0;



        vseq->general_level_idc = avctx->level * 3;



        vseq->intra_period = 0;

        vseq->intra_idr_period = 0;

        vseq->ip_period = 0;



        vseq->pic_width_in_luma_samples  = ctx->aligned_width;

        vseq->pic_height_in_luma_samples = ctx->aligned_height;



        vseq->seq_fields.bits.chroma_format_idc = 1; // 4:2:0.

        vseq->seq_fields.bits.separate_colour_plane_flag = 0;

        vseq->seq_fields.bits.bit_depth_luma_minus8 = 0; // 8-bit luma.

        vseq->seq_fields.bits.bit_depth_chroma_minus8 = 0; // 8-bit chroma.

        // Other misc flags all zero.



        // These have to come from the capabilities of the encoder.  We have

        // no way to query it, so just hardcode ones which worked for me...

        // CTB size from 8x8 to 32x32.

        vseq->log2_min_luma_coding_block_size_minus3 = 0;

        vseq->log2_diff_max_min_luma_coding_block_size = 2;

        // Transform size from 4x4 to 32x32.

        vseq->log2_min_transform_block_size_minus2 = 0;

        vseq->log2_diff_max_min_transform_block_size = 3;

        // Full transform hierarchy allowed (2-5).

        vseq->max_transform_hierarchy_depth_inter = 3;

        vseq->max_transform_hierarchy_depth_intra = 3;



        vseq->vui_parameters_present_flag = 0;



        vseq->bits_per_second = avctx->bit_rate;

        if (avctx->framerate.num > 0 && avctx->framerate.den > 0) {

            vseq->vui_num_units_in_tick = avctx->framerate.num;

            vseq->vui_time_scale        = avctx->framerate.den;

        } else {

            vseq->vui_num_units_in_tick = avctx->time_base.num;

            vseq->vui_time_scale        = avctx->time_base.den;

        }



        vseq->intra_period     = ctx->p_per_i * (ctx->b_per_p + 1);

        vseq->intra_idr_period = vseq->intra_period;

        vseq->ip_period        = ctx->b_per_p + 1;

    }



    {

        vpic->decoded_curr_pic.picture_id = VA_INVALID_ID;

        vpic->decoded_curr_pic.flags      = VA_PICTURE_HEVC_INVALID;



        for (i = 0; i < FF_ARRAY_ELEMS(vpic->reference_frames); i++) {

            vpic->reference_frames[i].picture_id = VA_INVALID_ID;

            vpic->reference_frames[i].flags      = VA_PICTURE_HEVC_INVALID;

        }



        vpic->collocated_ref_pic_index = 0xff;



        vpic->last_picture = 0;



        vpic->pic_init_qp = priv->fixed_qp_idr;



        vpic->diff_cu_qp_delta_depth = 0;

        vpic->pps_cb_qp_offset = 0;

        vpic->pps_cr_qp_offset = 0;



        // tiles_enabled_flag == 0, so ignore num_tile_(rows|columns)_minus1.



        vpic->log2_parallel_merge_level_minus2 = 0;



        // No limit on size.

        vpic->ctu_max_bitsize_allowed = 0;



        vpic->num_ref_idx_l0_default_active_minus1 = 0;

        vpic->num_ref_idx_l1_default_active_minus1 = 0;



        vpic->slice_pic_parameter_set_id = 0;



        vpic->pic_fields.bits.screen_content_flag = 0;

        vpic->pic_fields.bits.enable_gpu_weighted_prediction = 0;



        // Per-CU QP changes are required for non-constant-QP modes.

        vpic->pic_fields.bits.cu_qp_delta_enabled_flag =

            ctx->va_rc_mode != VA_RC_CQP;

    }



    {

        mseq->video_parameter_set_id = 5;

        mseq->seq_parameter_set_id = 5;



        mseq->vps_max_layers_minus1 = 0;

        mseq->vps_max_sub_layers_minus1 = 0;

        mseq->vps_temporal_id_nesting_flag = 1;

        mseq->sps_max_sub_layers_minus1 = 0;

        mseq->sps_temporal_id_nesting_flag = 1;



        for (i = 0; i < 32; i++) {

            mseq->general_profile_compatibility_flag[i] =

                (i == vseq->general_profile_idc);

        }



        mseq->general_progressive_source_flag    = 1;

        mseq->general_interlaced_source_flag     = 0;

        mseq->general_non_packed_constraint_flag = 0;

        mseq->general_frame_only_constraint_flag = 1;

        mseq->general_inbld_flag = 0;



        mseq->log2_max_pic_order_cnt_lsb_minus4 = 8;

        mseq->vps_sub_layer_ordering_info_present_flag = 0;

        mseq->vps_max_dec_pic_buffering_minus1[0] = 1;

        mseq->vps_max_num_reorder_pics[0]         = ctx->b_per_p;

        mseq->vps_max_latency_increase_plus1[0]   = 0;

        mseq->sps_sub_layer_ordering_info_present_flag = 0;

        mseq->sps_max_dec_pic_buffering_minus1[0] = 1;

        mseq->sps_max_num_reorder_pics[0]         = ctx->b_per_p;

        mseq->sps_max_latency_increase_plus1[0]   = 0;



        mseq->vps_timing_info_present_flag = 1;

        mseq->vps_num_units_in_tick = avctx->time_base.num;

        mseq->vps_time_scale        = avctx->time_base.den;

        mseq->vps_poc_proportional_to_timing_flag = 1;

        mseq->vps_num_ticks_poc_diff_minus1 = 0;



        if (ctx->input_width  != ctx->aligned_width ||

            ctx->input_height != ctx->aligned_height) {

            mseq->conformance_window_flag = 1;

            mseq->conf_win_left_offset   = 0;

            mseq->conf_win_right_offset  =

                (ctx->aligned_width - ctx->input_width) / 2;

            mseq->conf_win_top_offset    = 0;

            mseq->conf_win_bottom_offset =

                (ctx->aligned_height - ctx->input_height) / 2;

        } else {

            mseq->conformance_window_flag = 0;

        }



        mseq->num_short_term_ref_pic_sets = 0;

        // STRPSs should ideally be here rather than repeated in each slice.



        mseq->vui_parameters_present_flag = 1;

        if (avctx->sample_aspect_ratio.num != 0) {

            mseq->aspect_ratio_info_present_flag = 1;

            if (avctx->sample_aspect_ratio.num ==

                avctx->sample_aspect_ratio.den) {

                mseq->aspect_ratio_idc = 1;

            } else {

                mseq->aspect_ratio_idc = 255; // Extended SAR.

                mseq->sar_width  = avctx->sample_aspect_ratio.num;

                mseq->sar_height = avctx->sample_aspect_ratio.den;

            }

        }

        if (1) {

            // Should this be conditional on some of these being set?

            mseq->video_signal_type_present_flag = 1;

            mseq->video_format = 5; // Unspecified.

            mseq->video_full_range_flag = 0;

            mseq->colour_description_present_flag = 1;

            mseq->colour_primaries = avctx->color_primaries;

            mseq->transfer_characteristics = avctx->color_trc;

            mseq->matrix_coeffs = avctx->colorspace;

        }

    }



    return 0;

}
