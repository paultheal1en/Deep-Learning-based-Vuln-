static av_cold int nvenc_setup_device(AVCodecContext *avctx)

{

    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;



    CUresult cu_res;

    CUcontext cu_context_curr;



    switch (avctx->codec->id) {

    case AV_CODEC_ID_H264:

        ctx->init_encode_params.encodeGUID = NV_ENC_CODEC_H264_GUID;

        break;

    case AV_CODEC_ID_HEVC:

        ctx->init_encode_params.encodeGUID = NV_ENC_CODEC_HEVC_GUID;

        break;

    default:

        return AVERROR_BUG;

    }



    ctx->data_pix_fmt = avctx->pix_fmt;



#if CONFIG_CUDA

    if (avctx->pix_fmt == AV_PIX_FMT_CUDA) {

        AVHWFramesContext *frames_ctx;

        AVCUDADeviceContext *device_hwctx;



        if (!avctx->hw_frames_ctx) {

            av_log(avctx, AV_LOG_ERROR, "hw_frames_ctx must be set when using GPU frames as input\n");

            return AVERROR(EINVAL);

        }



        frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;

        device_hwctx = frames_ctx->device_ctx->hwctx;

        ctx->cu_context = device_hwctx->cuda_ctx;

        ctx->data_pix_fmt = frames_ctx->sw_format;

        return 0;

    }

#endif



    if (ctx->gpu >= dl_fn->nvenc_device_count) {

        av_log(avctx, AV_LOG_FATAL, "Requested GPU %d, but only %d GPUs are available!\n", ctx->gpu, dl_fn->nvenc_device_count);

        return AVERROR(EINVAL);

    }



    ctx->cu_context = NULL;

    cu_res = dl_fn->cu_ctx_create(&ctx->cu_context_internal, 4, dl_fn->nvenc_devices[ctx->gpu]); // CU_CTX_SCHED_BLOCKING_SYNC=4, avoid CPU spins



    if (cu_res != CUDA_SUCCESS) {

        av_log(avctx, AV_LOG_FATAL, "Failed creating CUDA context for NVENC: 0x%x\n", (int)cu_res);

        return AVERROR_EXTERNAL;

    }



    cu_res = dl_fn->cu_ctx_pop_current(&cu_context_curr);



    if (cu_res != CUDA_SUCCESS) {

        av_log(avctx, AV_LOG_FATAL, "Failed popping CUDA context: 0x%x\n", (int)cu_res);

        return AVERROR_EXTERNAL;

    }



    ctx->cu_context = ctx->cu_context_internal;



    return 0;

}
