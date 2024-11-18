int ff_nvdec_decode_init(AVCodecContext *avctx)

{

    NVDECContext *ctx = avctx->internal->hwaccel_priv_data;



    NVDECFramePool      *pool;

    AVHWFramesContext   *frames_ctx;

    const AVPixFmtDescriptor *sw_desc;



    CUVIDDECODECREATEINFO params = { 0 };



    int cuvid_codec_type, cuvid_chroma_format;

    int ret = 0;



    sw_desc = av_pix_fmt_desc_get(avctx->sw_pix_fmt);

    if (!sw_desc)

        return AVERROR_BUG;



    cuvid_codec_type = map_avcodec_id(avctx->codec_id);

    if (cuvid_codec_type < 0) {

        av_log(avctx, AV_LOG_ERROR, "Unsupported codec ID\n");

        return AVERROR_BUG;

    }



    cuvid_chroma_format = map_chroma_format(avctx->sw_pix_fmt);

    if (cuvid_chroma_format < 0) {

        av_log(avctx, AV_LOG_ERROR, "Unsupported chroma format\n");

        return AVERROR(ENOSYS);

    }



    if (!avctx->hw_frames_ctx) {

        ret = ff_decode_get_hw_frames_ctx(avctx, AV_HWDEVICE_TYPE_CUDA);

        if (ret < 0)

            return ret;

    }



    frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;



    params.ulWidth             = avctx->coded_width;

    params.ulHeight            = avctx->coded_height;

    params.ulTargetWidth       = avctx->coded_width;

    params.ulTargetHeight      = avctx->coded_height;

    params.bitDepthMinus8      = sw_desc->comp[0].depth - 8;

    params.OutputFormat        = params.bitDepthMinus8 ?

                                 cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;

    params.CodecType           = cuvid_codec_type;

    params.ChromaFormat        = cuvid_chroma_format;

    params.ulNumDecodeSurfaces = frames_ctx->initial_pool_size;

    params.ulNumOutputSurfaces = 1;



    ret = nvdec_decoder_create(&ctx->decoder_ref, frames_ctx->device_ref, &params, avctx);

    if (ret < 0)

        return ret;



    pool = av_mallocz(sizeof(*pool));

    if (!pool) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }

    pool->dpb_size = frames_ctx->initial_pool_size;



    ctx->decoder_pool = av_buffer_pool_init2(sizeof(int), pool,

                                             nvdec_decoder_frame_alloc, av_free);

    if (!ctx->decoder_pool) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    return 0;

fail:

    ff_nvdec_decode_uninit(avctx);

    return ret;

}
