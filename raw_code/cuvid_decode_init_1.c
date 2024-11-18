static av_cold int cuvid_decode_init(AVCodecContext *avctx)

{

    CuvidContext *ctx = avctx->priv_data;

    AVCUDADeviceContext *device_hwctx;

    AVHWDeviceContext *device_ctx;

    AVHWFramesContext *hwframe_ctx;

    CUVIDPARSERPARAMS cuparseinfo;

    CUVIDEOFORMATEX cuparse_ext;

    CUVIDSOURCEDATAPACKET seq_pkt;

    CUdevice device;

    CUcontext cuda_ctx = NULL;

    CUcontext dummy;

    const AVBitStreamFilter *bsf;

    int ret = 0;



    enum AVPixelFormat pix_fmts[3] = { AV_PIX_FMT_CUDA,

                                       AV_PIX_FMT_NV12,

                                       AV_PIX_FMT_NONE };



    ret = ff_get_format(avctx, pix_fmts);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "ff_get_format failed: %d\n", ret);

        return ret;

    }



    ctx->frame_queue = av_fifo_alloc(MAX_FRAME_COUNT * sizeof(CUVIDPARSERDISPINFO));

    if (!ctx->frame_queue) {

        ret = AVERROR(ENOMEM);

        goto error;

    }



    avctx->pix_fmt = ret;



    if (avctx->hw_frames_ctx) {

        ctx->hwframe = av_buffer_ref(avctx->hw_frames_ctx);

        if (!ctx->hwframe) {

            ret = AVERROR(ENOMEM);

            goto error;

        }



        hwframe_ctx = (AVHWFramesContext*)ctx->hwframe->data;



        ctx->hwdevice = av_buffer_ref(hwframe_ctx->device_ref);

        if (!ctx->hwdevice) {

            ret = AVERROR(ENOMEM);

            goto error;

        }



        device_ctx = hwframe_ctx->device_ctx;

        device_hwctx = device_ctx->hwctx;

        cuda_ctx = device_hwctx->cuda_ctx;

    } else {

        ctx->hwdevice = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);

        if (!ctx->hwdevice) {

            av_log(avctx, AV_LOG_ERROR, "Error allocating hwdevice\n");

            ret = AVERROR(ENOMEM);

            goto error;

        }



        ret = CHECK_CU(cuInit(0));

        if (ret < 0)

            goto error;



        ret = CHECK_CU(cuDeviceGet(&device, 0));

        if (ret < 0)

            goto error;



        ret = CHECK_CU(cuCtxCreate(&cuda_ctx, CU_CTX_SCHED_BLOCKING_SYNC, device));

        if (ret < 0)

            goto error;



        device_ctx = (AVHWDeviceContext*)ctx->hwdevice->data;

        device_ctx->free = cuvid_ctx_free;



        device_hwctx = device_ctx->hwctx;

        device_hwctx->cuda_ctx = cuda_ctx;



        ret = CHECK_CU(cuCtxPopCurrent(&dummy));

        if (ret < 0)

            goto error;



        ret = av_hwdevice_ctx_init(ctx->hwdevice);

        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "av_hwdevice_ctx_init failed\n");

            goto error;

        }



        ctx->hwframe = av_hwframe_ctx_alloc(ctx->hwdevice);

        if (!ctx->hwframe) {

            av_log(avctx, AV_LOG_ERROR, "av_hwframe_ctx_alloc failed\n");

            ret = AVERROR(ENOMEM);

            goto error;

        }

    }



    memset(&cuparseinfo, 0, sizeof(cuparseinfo));

    memset(&cuparse_ext, 0, sizeof(cuparse_ext));

    memset(&seq_pkt, 0, sizeof(seq_pkt));



    cuparseinfo.pExtVideoInfo = &cuparse_ext;



    switch (avctx->codec->id) {






#if CONFIG_H264_CUVID_DECODER

    case AV_CODEC_ID_H264:

        cuparseinfo.CodecType = cudaVideoCodec_H264;



#if CONFIG_HEVC_CUVID_DECODER

    case AV_CODEC_ID_HEVC:

        cuparseinfo.CodecType = cudaVideoCodec_HEVC;



#if CONFIG_MJPEG_CUVID_DECODER

    case AV_CODEC_ID_MJPEG:

        cuparseinfo.CodecType = cudaVideoCodec_JPEG;



#if CONFIG_MPEG1_CUVID_DECODER

    case AV_CODEC_ID_MPEG1VIDEO:

        cuparseinfo.CodecType = cudaVideoCodec_MPEG1;



#if CONFIG_MPEG2_CUVID_DECODER

    case AV_CODEC_ID_MPEG2VIDEO:

        cuparseinfo.CodecType = cudaVideoCodec_MPEG2;



#if CONFIG_MPEG4_CUVID_DECODER

    case AV_CODEC_ID_MPEG4:




#if CONFIG_VP8_CUVID_DECODER

    case AV_CODEC_ID_VP8:

        cuparseinfo.CodecType = cudaVideoCodec_VP8;



#if CONFIG_VP9_CUVID_DECODER

    case AV_CODEC_ID_VP9:

        cuparseinfo.CodecType = cudaVideoCodec_VP9;



#if CONFIG_VC1_CUVID_DECODER

    case AV_CODEC_ID_VC1:

        cuparseinfo.CodecType = cudaVideoCodec_VC1;



    default:

        av_log(avctx, AV_LOG_ERROR, "Invalid CUVID codec!\n");

        return AVERROR_BUG;

    }



    if (avctx->codec->id == AV_CODEC_ID_H264 || avctx->codec->id == AV_CODEC_ID_HEVC) {

        if (avctx->codec->id == AV_CODEC_ID_H264)

            bsf = av_bsf_get_by_name("h264_mp4toannexb");

        else

            bsf = av_bsf_get_by_name("hevc_mp4toannexb");



        if (!bsf) {

            ret = AVERROR_BSF_NOT_FOUND;

            goto error;

        }

        if (ret = av_bsf_alloc(bsf, &ctx->bsf)) {

            goto error;

        }

        if (((ret = avcodec_parameters_from_context(ctx->bsf->par_in, avctx)) < 0) || ((ret = av_bsf_init(ctx->bsf)) < 0)) {

            av_bsf_free(&ctx->bsf);

            goto error;

        }



        cuparse_ext.format.seqhdr_data_length = ctx->bsf->par_out->extradata_size;

        memcpy(cuparse_ext.raw_seqhdr_data,

               ctx->bsf->par_out->extradata,

               FFMIN(sizeof(cuparse_ext.raw_seqhdr_data), ctx->bsf->par_out->extradata_size));

    } else if (avctx->extradata_size > 0) {

        cuparse_ext.format.seqhdr_data_length = avctx->extradata_size;

        memcpy(cuparse_ext.raw_seqhdr_data,

               avctx->extradata,

               FFMIN(sizeof(cuparse_ext.raw_seqhdr_data), avctx->extradata_size));

    }



    cuparseinfo.ulMaxNumDecodeSurfaces = MAX_FRAME_COUNT;

    cuparseinfo.ulMaxDisplayDelay = 4;

    cuparseinfo.pUserData = avctx;

    cuparseinfo.pfnSequenceCallback = cuvid_handle_video_sequence;

    cuparseinfo.pfnDecodePicture = cuvid_handle_picture_decode;

    cuparseinfo.pfnDisplayPicture = cuvid_handle_picture_display;



    ret = CHECK_CU(cuCtxPushCurrent(cuda_ctx));

    if (ret < 0)

        goto error;



    ret = cuvid_test_dummy_decoder(avctx, &cuparseinfo);

    if (ret < 0)

        goto error;



    ret = CHECK_CU(cuvidCreateVideoParser(&ctx->cuparser, &cuparseinfo));

    if (ret < 0)

        goto error;



    seq_pkt.payload = cuparse_ext.raw_seqhdr_data;

    seq_pkt.payload_size = cuparse_ext.format.seqhdr_data_length;



    if (seq_pkt.payload && seq_pkt.payload_size) {

        ret = CHECK_CU(cuvidParseVideoData(ctx->cuparser, &seq_pkt));

        if (ret < 0)

            goto error;

    }



    ret = CHECK_CU(cuCtxPopCurrent(&dummy));

    if (ret < 0)

        goto error;



    return 0;



error:

    cuvid_decode_end(avctx);

    return ret;

}