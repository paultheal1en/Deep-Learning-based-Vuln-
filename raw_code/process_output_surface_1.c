static int process_output_surface(AVCodecContext *avctx, AVPacket *pkt, NvencSurface *tmpoutsurf)

{

    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;

    NV_ENCODE_API_FUNCTION_LIST *p_nvenc = &dl_fn->nvenc_funcs;



    uint32_t slice_mode_data;

    uint32_t *slice_offsets;

    NV_ENC_LOCK_BITSTREAM lock_params = { 0 };

    NVENCSTATUS nv_status;

    int res = 0;



    enum AVPictureType pict_type;



    switch (avctx->codec->id) {

    case AV_CODEC_ID_H264:

      slice_mode_data = ctx->encode_config.encodeCodecConfig.h264Config.sliceModeData;

      break;

    case AV_CODEC_ID_H265:

      slice_mode_data = ctx->encode_config.encodeCodecConfig.hevcConfig.sliceModeData;

      break;

    default:

      av_log(avctx, AV_LOG_ERROR, "Unknown codec name\n");

      res = AVERROR(EINVAL);

      goto error;

    }

    slice_offsets = av_mallocz(slice_mode_data * sizeof(*slice_offsets));



    if (!slice_offsets)

        goto error;



    lock_params.version = NV_ENC_LOCK_BITSTREAM_VER;



    lock_params.doNotWait = 0;

    lock_params.outputBitstream = tmpoutsurf->output_surface;

    lock_params.sliceOffsets = slice_offsets;



    nv_status = p_nvenc->nvEncLockBitstream(ctx->nvencoder, &lock_params);

    if (nv_status != NV_ENC_SUCCESS) {

        res = nvenc_print_error(avctx, nv_status, "Failed locking bitstream buffer");

        goto error;

    }



    if (res = ff_alloc_packet2(avctx, pkt, lock_params.bitstreamSizeInBytes,0)) {

        p_nvenc->nvEncUnlockBitstream(ctx->nvencoder, tmpoutsurf->output_surface);

        goto error;

    }



    memcpy(pkt->data, lock_params.bitstreamBufferPtr, lock_params.bitstreamSizeInBytes);



    nv_status = p_nvenc->nvEncUnlockBitstream(ctx->nvencoder, tmpoutsurf->output_surface);

    if (nv_status != NV_ENC_SUCCESS)

        nvenc_print_error(avctx, nv_status, "Failed unlocking bitstream buffer, expect the gates of mordor to open");





    if (avctx->pix_fmt == AV_PIX_FMT_CUDA) {

        p_nvenc->nvEncUnmapInputResource(ctx->nvencoder, tmpoutsurf->in_map.mappedResource);

        av_frame_unref(tmpoutsurf->in_ref);

        ctx->registered_frames[tmpoutsurf->reg_idx].mapped = 0;



        tmpoutsurf->input_surface = NULL;

    }



    switch (lock_params.pictureType) {

    case NV_ENC_PIC_TYPE_IDR:

        pkt->flags |= AV_PKT_FLAG_KEY;

    case NV_ENC_PIC_TYPE_I:

        pict_type = AV_PICTURE_TYPE_I;

        break;

    case NV_ENC_PIC_TYPE_P:

        pict_type = AV_PICTURE_TYPE_P;

        break;

    case NV_ENC_PIC_TYPE_B:

        pict_type = AV_PICTURE_TYPE_B;

        break;

    case NV_ENC_PIC_TYPE_BI:

        pict_type = AV_PICTURE_TYPE_BI;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unknown picture type encountered, expect the output to be broken.\n");

        av_log(avctx, AV_LOG_ERROR, "Please report this error and include as much information on how to reproduce it as possible.\n");

        res = AVERROR_EXTERNAL;

        goto error;

    }



#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    avctx->coded_frame->pict_type = pict_type;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    ff_side_data_set_encoder_stats(pkt,

        (lock_params.frameAvgQP - 1) * FF_QP2LAMBDA, NULL, 0, pict_type);



    res = nvenc_set_timestamp(avctx, &lock_params, pkt);

    if (res < 0)

        goto error2;



    av_free(slice_offsets);



    return 0;



error:

    timestamp_queue_dequeue(ctx->timestamp_list);



error2:

    av_free(slice_offsets);



    return res;

}
