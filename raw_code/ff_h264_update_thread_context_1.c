int ff_h264_update_thread_context(AVCodecContext *dst,

                                  const AVCodecContext *src)

{

    H264Context *h = dst->priv_data, *h1 = src->priv_data;

    int inited = h->context_initialized, err = 0;

    int context_reinitialized = 0;

    int i, ret;



    if (dst == src)

        return 0;



    if (inited &&

        (h->width                 != h1->width                 ||

         h->height                != h1->height                ||

         h->mb_width              != h1->mb_width              ||

         h->mb_height             != h1->mb_height             ||

         h->sps.bit_depth_luma    != h1->sps.bit_depth_luma    ||

         h->sps.chroma_format_idc != h1->sps.chroma_format_idc ||

         h->sps.colorspace        != h1->sps.colorspace)) {



        /* set bits_per_raw_sample to the previous value. the check for changed

         * bit depth in h264_set_parameter_from_sps() uses it and sets it to

         * the current value */

        h->avctx->bits_per_raw_sample = h->sps.bit_depth_luma;



        h->width     = h1->width;

        h->height    = h1->height;

        h->mb_height = h1->mb_height;

        h->mb_width  = h1->mb_width;

        h->mb_num    = h1->mb_num;

        h->mb_stride = h1->mb_stride;

        h->b_stride  = h1->b_stride;

        // SPS/PPS

        if ((ret = copy_parameter_set((void **)h->sps_buffers,

                                      (void **)h1->sps_buffers,

                                      MAX_SPS_COUNT, sizeof(SPS))) < 0)

            return ret;

        h->sps = h1->sps;

        if ((ret = copy_parameter_set((void **)h->pps_buffers,

                                      (void **)h1->pps_buffers,

                                      MAX_PPS_COUNT, sizeof(PPS))) < 0)

            return ret;

        h->pps = h1->pps;



        if ((err = h264_slice_header_init(h, 1)) < 0) {

            av_log(h->avctx, AV_LOG_ERROR, "h264_slice_header_init() failed\n");

            return err;

        }

        context_reinitialized = 1;



#if 0

        h264_set_parameter_from_sps(h);

        //Note we set context_reinitialized which will cause h264_set_parameter_from_sps to be reexecuted

        h->cur_chroma_format_idc = h1->cur_chroma_format_idc;

#endif

    }



    /* copy block_offset since frame_start may not be called */

    memcpy(h->block_offset, h1->block_offset, sizeof(h->block_offset));



    if (!inited) {

        H264SliceContext *orig_slice_ctx = h->slice_ctx;



        for (i = 0; i < MAX_SPS_COUNT; i++)

            av_freep(h->sps_buffers + i);



        for (i = 0; i < MAX_PPS_COUNT; i++)

            av_freep(h->pps_buffers + i);



        ff_h264_unref_picture(h, &h->last_pic_for_ec);

        memcpy(h, h1, sizeof(H264Context));



        memset(h->sps_buffers, 0, sizeof(h->sps_buffers));

        memset(h->pps_buffers, 0, sizeof(h->pps_buffers));



        memset(&h->cur_pic, 0, sizeof(h->cur_pic));

        memset(&h->last_pic_for_ec, 0, sizeof(h->last_pic_for_ec));



        h->slice_ctx = orig_slice_ctx;



        memset(&h->slice_ctx[0].er,         0, sizeof(h->slice_ctx[0].er));

        memset(&h->slice_ctx[0].mb,         0, sizeof(h->slice_ctx[0].mb));

        memset(&h->slice_ctx[0].mb_luma_dc, 0, sizeof(h->slice_ctx[0].mb_luma_dc));

        memset(&h->slice_ctx[0].mb_padding, 0, sizeof(h->slice_ctx[0].mb_padding));



        h->avctx             = dst;

        h->DPB               = NULL;

        h->qscale_table_pool = NULL;

        h->mb_type_pool      = NULL;

        h->ref_index_pool    = NULL;

        h->motion_val_pool   = NULL;

        h->intra4x4_pred_mode= NULL;

        h->non_zero_count    = NULL;

        h->slice_table_base  = NULL;

        h->slice_table       = NULL;

        h->cbp_table         = NULL;

        h->chroma_pred_mode_table = NULL;

        memset(h->mvd_table, 0, sizeof(h->mvd_table));

        h->direct_table      = NULL;

        h->list_counts       = NULL;

        h->mb2b_xy           = NULL;

        h->mb2br_xy          = NULL;



        if (h1->context_initialized) {

        h->context_initialized = 0;



        memset(&h->cur_pic, 0, sizeof(h->cur_pic));

        av_frame_unref(&h->cur_pic.f);

        h->cur_pic.tf.f = &h->cur_pic.f;



        ret = ff_h264_alloc_tables(h);

        if (ret < 0) {

            av_log(dst, AV_LOG_ERROR, "Could not allocate memory\n");

            return ret;

        }

        ret = ff_h264_slice_context_init(h, &h->slice_ctx[0]);

        if (ret < 0) {

            av_log(dst, AV_LOG_ERROR, "context_init() failed.\n");

            return ret;

        }

        }



        h->context_initialized = h1->context_initialized;

    }



    h->avctx->coded_height  = h1->avctx->coded_height;

    h->avctx->coded_width   = h1->avctx->coded_width;

    h->avctx->width         = h1->avctx->width;

    h->avctx->height        = h1->avctx->height;

    h->coded_picture_number = h1->coded_picture_number;

    h->first_field          = h1->first_field;

    h->picture_structure    = h1->picture_structure;

    h->droppable            = h1->droppable;

    h->low_delay            = h1->low_delay;



    for (i = 0; h->DPB && i < H264_MAX_PICTURE_COUNT; i++) {

        ff_h264_unref_picture(h, &h->DPB[i]);

        if (h1->DPB && h1->DPB[i].f.buf[0] &&

            (ret = ff_h264_ref_picture(h, &h->DPB[i], &h1->DPB[i])) < 0)

            return ret;

    }



    h->cur_pic_ptr = REBASE_PICTURE(h1->cur_pic_ptr, h, h1);

    ff_h264_unref_picture(h, &h->cur_pic);

    if (h1->cur_pic.f.buf[0]) {

        ret = ff_h264_ref_picture(h, &h->cur_pic, &h1->cur_pic);

        if (ret < 0)

            return ret;

    }



    h->workaround_bugs = h1->workaround_bugs;

    h->low_delay       = h1->low_delay;

    h->droppable       = h1->droppable;



    // extradata/NAL handling

    h->is_avc = h1->is_avc;



    // SPS/PPS

    if ((ret = copy_parameter_set((void **)h->sps_buffers,

                                  (void **)h1->sps_buffers,

                                  MAX_SPS_COUNT, sizeof(SPS))) < 0)

        return ret;

    h->sps = h1->sps;

    if ((ret = copy_parameter_set((void **)h->pps_buffers,

                                  (void **)h1->pps_buffers,

                                  MAX_PPS_COUNT, sizeof(PPS))) < 0)

        return ret;

    h->pps = h1->pps;



    // Dequantization matrices

    // FIXME these are big - can they be only copied when PPS changes?

    copy_fields(h, h1, dequant4_buffer, dequant4_coeff);



    for (i = 0; i < 6; i++)

        h->dequant4_coeff[i] = h->dequant4_buffer[0] +

                               (h1->dequant4_coeff[i] - h1->dequant4_buffer[0]);



    for (i = 0; i < 6; i++)

        h->dequant8_coeff[i] = h->dequant8_buffer[0] +

                               (h1->dequant8_coeff[i] - h1->dequant8_buffer[0]);



    h->dequant_coeff_pps = h1->dequant_coeff_pps;



    // POC timing

    copy_fields(h, h1, poc_lsb, default_ref_list);



    // reference lists

    copy_fields(h, h1, short_ref, current_slice);



    copy_picture_range(h->short_ref, h1->short_ref, 32, h, h1);

    copy_picture_range(h->long_ref, h1->long_ref, 32, h, h1);

    copy_picture_range(h->delayed_pic, h1->delayed_pic,

                       MAX_DELAYED_PIC_COUNT + 2, h, h1);



    h->frame_recovered       = h1->frame_recovered;



    if (context_reinitialized)

        ff_h264_set_parameter_from_sps(h);



    if (!h->cur_pic_ptr)

        return 0;



    if (!h->droppable) {

        err = ff_h264_execute_ref_pic_marking(h, h->mmco, h->mmco_index);

        h->prev_poc_msb = h->poc_msb;

        h->prev_poc_lsb = h->poc_lsb;

    }

    h->prev_frame_num_offset = h->frame_num_offset;

    h->prev_frame_num        = h->frame_num;

    h->outputed_poc          = h->next_outputed_poc;



    h->recovery_frame        = h1->recovery_frame;



    return err;

}
