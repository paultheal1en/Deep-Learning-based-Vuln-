static int decode_update_thread_context(AVCodecContext *dst,

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



        av_freep(&h->bipred_scratchpad);



        h->width     = h1->width;

        h->height    = h1->height;

        h->mb_height = h1->mb_height;

        h->mb_width  = h1->mb_width;

        h->mb_num    = h1->mb_num;

        h->mb_stride = h1->mb_stride;

        h->b_stride  = h1->b_stride;

        // SPS/PPS

        copy_parameter_set((void **)h->sps_buffers, (void **)h1->sps_buffers,

                        MAX_SPS_COUNT, sizeof(SPS));

        h->sps = h1->sps;

        copy_parameter_set((void **)h->pps_buffers, (void **)h1->pps_buffers,

                        MAX_PPS_COUNT, sizeof(PPS));

        h->pps = h1->pps;



        if ((err = h264_slice_header_init(h, 1)) < 0) {

            av_log(h->avctx, AV_LOG_ERROR, "h264_slice_header_init() failed");

            return err;

        }

        context_reinitialized = 1;



#if 0

        h264_set_parameter_from_sps(h);

        //Note we set context_reinitialized which will cause h264_set_parameter_from_sps to be reexecuted

        h->cur_chroma_format_idc = h1->cur_chroma_format_idc;

#endif

    }

    /* update linesize on resize for h264. The h264 decoder doesn't

     * necessarily call ff_MPV_frame_start in the new thread */

    h->linesize   = h1->linesize;

    h->uvlinesize = h1->uvlinesize;



    /* copy block_offset since frame_start may not be called */

    memcpy(h->block_offset, h1->block_offset, sizeof(h->block_offset));



    if (!inited) {

        for (i = 0; i < MAX_SPS_COUNT; i++)

            av_freep(h->sps_buffers + i);



        for (i = 0; i < MAX_PPS_COUNT; i++)

            av_freep(h->pps_buffers + i);



        memcpy(h, h1, offsetof(H264Context, intra_pcm_ptr));

        memcpy(&h->cabac, &h1->cabac,

               sizeof(H264Context) - offsetof(H264Context, cabac));

        av_assert0((void*)&h->cabac == &h->mb_padding + 1);



        memset(h->sps_buffers, 0, sizeof(h->sps_buffers));

        memset(h->pps_buffers, 0, sizeof(h->pps_buffers));



        memset(&h->er, 0, sizeof(h->er));

        memset(&h->me, 0, sizeof(h->me));

        memset(&h->mb, 0, sizeof(h->mb));

        memset(&h->mb_luma_dc, 0, sizeof(h->mb_luma_dc));

        memset(&h->mb_padding, 0, sizeof(h->mb_padding));



        h->avctx             = dst;

        h->DPB               = NULL;

        h->qscale_table_pool = NULL;

        h->mb_type_pool      = NULL;

        h->ref_index_pool    = NULL;

        h->motion_val_pool   = NULL;



        if (h1->context_initialized) {

        h->context_initialized = 0;



        memset(&h->cur_pic, 0, sizeof(h->cur_pic));

        avcodec_get_frame_defaults(&h->cur_pic.f);

        h->cur_pic.tf.f = &h->cur_pic.f;



        ret = ff_h264_alloc_tables(h);

        if (ret < 0) {

            av_log(dst, AV_LOG_ERROR, "Could not allocate memory for h264\n");

            return ret;

        }

        ret = context_init(h);

        if (ret < 0) {

            av_log(dst, AV_LOG_ERROR, "context_init() failed.\n");

            return ret;

        }

        }



        for (i = 0; i < 2; i++) {

            h->rbsp_buffer[i]      = NULL;

            h->rbsp_buffer_size[i] = 0;

        }

        h->bipred_scratchpad = NULL;

        h->edge_emu_buffer   = NULL;



        h->thread_context[0] = h;

        h->context_initialized = h1->context_initialized;

    }



    h->avctx->coded_height  = h1->avctx->coded_height;

    h->avctx->coded_width   = h1->avctx->coded_width;

    h->avctx->width         = h1->avctx->width;

    h->avctx->height        = h1->avctx->height;

    h->coded_picture_number = h1->coded_picture_number;

    h->first_field          = h1->first_field;

    h->picture_structure    = h1->picture_structure;

    h->qscale               = h1->qscale;

    h->droppable            = h1->droppable;

    h->data_partitioning    = h1->data_partitioning;

    h->low_delay            = h1->low_delay;



    for (i = 0; h->DPB && i < MAX_PICTURE_COUNT; i++) {

        unref_picture(h, &h->DPB[i]);

        if (h1->DPB[i].f.data[0] &&

            (ret = ref_picture(h, &h->DPB[i], &h1->DPB[i])) < 0)

            return ret;

    }



    h->cur_pic_ptr = REBASE_PICTURE(h1->cur_pic_ptr, h, h1);

    unref_picture(h, &h->cur_pic);

    if (h1->cur_pic.f.buf[0] && (ret = ref_picture(h, &h->cur_pic, &h1->cur_pic)) < 0)

        return ret;



    h->workaround_bugs = h1->workaround_bugs;

    h->low_delay       = h1->low_delay;

    h->droppable       = h1->droppable;



    // extradata/NAL handling

    h->is_avc = h1->is_avc;



    // SPS/PPS

    copy_parameter_set((void **)h->sps_buffers, (void **)h1->sps_buffers,

                       MAX_SPS_COUNT, sizeof(SPS));

    h->sps = h1->sps;

    copy_parameter_set((void **)h->pps_buffers, (void **)h1->pps_buffers,

                       MAX_PPS_COUNT, sizeof(PPS));

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

    copy_fields(h, h1, poc_lsb, redundant_pic_count);



    // reference lists

    copy_fields(h, h1, short_ref, cabac_init_idc);



    copy_picture_range(h->short_ref, h1->short_ref, 32, h, h1);

    copy_picture_range(h->long_ref, h1->long_ref, 32, h, h1);

    copy_picture_range(h->delayed_pic, h1->delayed_pic,

                       MAX_DELAYED_PIC_COUNT + 2, h, h1);



    h->sync            = h1->sync;



    if (context_reinitialized)

        h264_set_parameter_from_sps(h);



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



    return err;

}
