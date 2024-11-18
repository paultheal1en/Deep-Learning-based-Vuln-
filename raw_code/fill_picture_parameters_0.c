static void fill_picture_parameters(const AVCodecContext *avctx, AVDXVAContext *ctx, const H264Context *h,

                                    DXVA_PicParams_H264 *pp)

{

    const H264Picture *current_picture = h->cur_pic_ptr;

    const SPS *sps = h->ps.sps;

    const PPS *pps = h->ps.pps;

    int i, j;



    memset(pp, 0, sizeof(*pp));

    /* Configure current picture */

    fill_picture_entry(&pp->CurrPic,

                       ff_dxva2_get_surface_index(avctx, ctx, current_picture->f),

                       h->picture_structure == PICT_BOTTOM_FIELD);

    /* Configure the set of references */

    pp->UsedForReferenceFlags  = 0;

    pp->NonExistingFrameFlags  = 0;

    for (i = 0, j = 0; i < FF_ARRAY_ELEMS(pp->RefFrameList); i++) {

        const H264Picture *r;

        if (j < h->short_ref_count) {

            r = h->short_ref[j++];

        } else {

            r = NULL;

            while (!r && j < h->short_ref_count + 16)

                r = h->long_ref[j++ - h->short_ref_count];

        }

        if (r) {

            fill_picture_entry(&pp->RefFrameList[i],

                               ff_dxva2_get_surface_index(avctx, ctx, r->f),

                               r->long_ref != 0);



            if ((r->reference & PICT_TOP_FIELD) && r->field_poc[0] != INT_MAX)

                pp->FieldOrderCntList[i][0] = r->field_poc[0];

            if ((r->reference & PICT_BOTTOM_FIELD) && r->field_poc[1] != INT_MAX)

                pp->FieldOrderCntList[i][1] = r->field_poc[1];



            pp->FrameNumList[i] = r->long_ref ? r->pic_id : r->frame_num;

            if (r->reference & PICT_TOP_FIELD)

                pp->UsedForReferenceFlags |= 1 << (2*i + 0);

            if (r->reference & PICT_BOTTOM_FIELD)

                pp->UsedForReferenceFlags |= 1 << (2*i + 1);

        } else {

            pp->RefFrameList[i].bPicEntry = 0xff;

            pp->FieldOrderCntList[i][0]   = 0;

            pp->FieldOrderCntList[i][1]   = 0;

            pp->FrameNumList[i]           = 0;

        }

    }



    pp->wFrameWidthInMbsMinus1        = h->mb_width  - 1;

    pp->wFrameHeightInMbsMinus1       = h->mb_height - 1;

    pp->num_ref_frames                = sps->ref_frame_count;



    pp->wBitFields                    = ((h->picture_structure != PICT_FRAME) <<  0) |

                                        ((sps->mb_aff &&

                                        (h->picture_structure == PICT_FRAME)) <<  1) |

                                        (sps->residual_color_transform_flag   <<  2) |

                                        /* sp_for_switch_flag (not implemented by Libav) */

                                        (0                                    <<  3) |

                                        (sps->chroma_format_idc               <<  4) |

                                        ((h->nal_ref_idc != 0)                <<  6) |

                                        (pps->constrained_intra_pred          <<  7) |

                                        (pps->weighted_pred                   <<  8) |

                                        (pps->weighted_bipred_idc             <<  9) |

                                        /* MbsConsecutiveFlag */

                                        (1                                    << 11) |

                                        (sps->frame_mbs_only_flag             << 12) |

                                        (pps->transform_8x8_mode              << 13) |

                                        ((sps->level_idc >= 31)               << 14) |

                                        /* IntraPicFlag (Modified if we detect a non

                                         * intra slice in dxva2_h264_decode_slice) */

                                        (1                                    << 15);



    pp->bit_depth_luma_minus8         = sps->bit_depth_luma - 8;

    pp->bit_depth_chroma_minus8       = sps->bit_depth_chroma - 8;

    if (DXVA_CONTEXT_WORKAROUND(avctx, ctx) & FF_DXVA2_WORKAROUND_SCALING_LIST_ZIGZAG)

        pp->Reserved16Bits            = 0;

    else if (DXVA_CONTEXT_WORKAROUND(avctx, ctx) & FF_DXVA2_WORKAROUND_INTEL_CLEARVIDEO)

        pp->Reserved16Bits            = 0x34c;

    else

        pp->Reserved16Bits            = 3; /* FIXME is there a way to detect the right mode ? */

    pp->StatusReportFeedbackNumber    = 1 + DXVA_CONTEXT_REPORT_ID(avctx, ctx)++;

    pp->CurrFieldOrderCnt[0] = 0;

    if ((h->picture_structure & PICT_TOP_FIELD) &&

        current_picture->field_poc[0] != INT_MAX)

        pp->CurrFieldOrderCnt[0] = current_picture->field_poc[0];

    pp->CurrFieldOrderCnt[1] = 0;

    if ((h->picture_structure & PICT_BOTTOM_FIELD) &&

        current_picture->field_poc[1] != INT_MAX)

        pp->CurrFieldOrderCnt[1] = current_picture->field_poc[1];

    pp->pic_init_qs_minus26           = pps->init_qs - 26;

    pp->chroma_qp_index_offset        = pps->chroma_qp_index_offset[0];

    pp->second_chroma_qp_index_offset = pps->chroma_qp_index_offset[1];

    pp->ContinuationFlag              = 1;

    pp->pic_init_qp_minus26           = pps->init_qp - 26;

    pp->num_ref_idx_l0_active_minus1  = pps->ref_count[0] - 1;

    pp->num_ref_idx_l1_active_minus1  = pps->ref_count[1] - 1;

    pp->Reserved8BitsA                = 0;

    pp->frame_num                     = h->frame_num;

    pp->log2_max_frame_num_minus4     = sps->log2_max_frame_num - 4;

    pp->pic_order_cnt_type            = sps->poc_type;

    if (sps->poc_type == 0)

        pp->log2_max_pic_order_cnt_lsb_minus4 = sps->log2_max_poc_lsb - 4;

    else if (sps->poc_type == 1)

        pp->delta_pic_order_always_zero_flag = sps->delta_pic_order_always_zero_flag;

    pp->direct_8x8_inference_flag     = sps->direct_8x8_inference_flag;

    pp->entropy_coding_mode_flag      = pps->cabac;

    pp->pic_order_present_flag        = pps->pic_order_present;

    pp->num_slice_groups_minus1       = pps->slice_group_count - 1;

    pp->slice_group_map_type          = pps->mb_slice_group_map_type;

    pp->deblocking_filter_control_present_flag = pps->deblocking_filter_parameters_present;

    pp->redundant_pic_cnt_present_flag= pps->redundant_pic_cnt_present;

    pp->Reserved8BitsB                = 0;

    pp->slice_group_change_rate_minus1= 0;  /* XXX not implemented by Libav */

    //pp->SliceGroupMap[810];               /* XXX not implemented by Libav */

}
