av_cold int ff_h264_decode_init(AVCodecContext *avctx)

{

    H264Context *h = avctx->priv_data;

    int i;

    int ret;



    h->avctx = avctx;



    h->bit_depth_luma    = 8;

    h->chroma_format_idc = 1;



    h->cur_chroma_format_idc = 1;



    ff_h264dsp_init(&h->h264dsp, 8, 1);

    av_assert0(h->sps.bit_depth_chroma == 0);

    ff_h264chroma_init(&h->h264chroma, h->sps.bit_depth_chroma);

    ff_h264qpel_init(&h->h264qpel, 8);

    ff_h264_pred_init(&h->hpc, h->avctx->codec_id, 8, 1);



    h->dequant_coeff_pps = -1;

    h->current_sps_id = -1;



    /* needed so that IDCT permutation is known early */

    ff_videodsp_init(&h->vdsp, 8);



    memset(h->pps.scaling_matrix4, 16, 6 * 16 * sizeof(uint8_t));

    memset(h->pps.scaling_matrix8, 16, 2 * 64 * sizeof(uint8_t));



    h->picture_structure   = PICT_FRAME;

    h->slice_context_count = 1;

    h->workaround_bugs     = avctx->workaround_bugs;

    h->flags               = avctx->flags;



    /* set defaults */

    // s->decode_mb = ff_h263_decode_mb;

    if (!avctx->has_b_frames)

        h->low_delay = 1;



    avctx->chroma_sample_location = AVCHROMA_LOC_LEFT;



    ff_h264_decode_init_vlc();



    ff_init_cabac_states();



    h->pixel_shift        = 0;

    h->cur_bit_depth_luma =

    h->sps.bit_depth_luma = avctx->bits_per_raw_sample = 8;



    h->nb_slice_ctx = (avctx->active_thread_type & FF_THREAD_SLICE) ?  H264_MAX_THREADS : 1;

    h->slice_ctx = av_mallocz_array(h->nb_slice_ctx, sizeof(*h->slice_ctx));

    if (!h->slice_ctx) {

        h->nb_slice_ctx = 0;

        return AVERROR(ENOMEM);

    }



    for (i = 0; i < h->nb_slice_ctx; i++)

        h->slice_ctx[i].h264 = h;



    h->outputed_poc      = h->next_outputed_poc = INT_MIN;

    for (i = 0; i < MAX_DELAYED_PIC_COUNT; i++)

        h->last_pocs[i] = INT_MIN;

    h->prev_poc_msb = 1 << 16;

    h->prev_frame_num = -1;

    h->x264_build   = -1;

    h->sei_fpa.frame_packing_arrangement_cancel_flag = -1;

    ff_h264_reset_sei(h);

    if (avctx->codec_id == AV_CODEC_ID_H264) {

        if (avctx->ticks_per_frame == 1) {

            if(h->avctx->time_base.den < INT_MAX/2) {

                h->avctx->time_base.den *= 2;

            } else

                h->avctx->time_base.num /= 2;

        }

        avctx->ticks_per_frame = 2;

    }



    if (avctx->extradata_size > 0 && avctx->extradata) {

        ret = ff_h264_decode_extradata(h, avctx->extradata, avctx->extradata_size);

        if (ret < 0) {

            ff_h264_free_context(h);

            return ret;

        }

    }



    if (h->sps.bitstream_restriction_flag &&

        h->avctx->has_b_frames < h->sps.num_reorder_frames) {

        h->avctx->has_b_frames = h->sps.num_reorder_frames;

        h->low_delay           = 0;

    }



    avctx->internal->allocate_progress = 1;



    ff_h264_flush_change(h);



    if (h->enable_er) {

        av_log(avctx, AV_LOG_WARNING,

               "Error resilience is enabled. It is unsafe and unsupported and may crash. "

               "Use it at your own risk\n");

    }



    return 0;

}
