static int qsv_decode_init(AVCodecContext *avctx, QSVContext *q)

{

    const AVPixFmtDescriptor *desc;

    mfxSession session = NULL;

    int iopattern = 0;

    mfxVideoParam param = { { 0 } };

    int frame_width  = avctx->coded_width;

    int frame_height = avctx->coded_height;

    int ret;



    desc = av_pix_fmt_desc_get(avctx->sw_pix_fmt);

    if (!desc)

        return AVERROR_BUG;



    if (!q->async_fifo) {

        q->async_fifo = av_fifo_alloc((1 + q->async_depth) *

                                      (sizeof(mfxSyncPoint*) + sizeof(QSVFrame*)));

        if (!q->async_fifo)

            return AVERROR(ENOMEM);

    }



    if (avctx->pix_fmt == AV_PIX_FMT_QSV && avctx->hwaccel_context) {

        AVQSVContext *user_ctx = avctx->hwaccel_context;

        session           = user_ctx->session;

        iopattern         = user_ctx->iopattern;

        q->ext_buffers    = user_ctx->ext_buffers;

        q->nb_ext_buffers = user_ctx->nb_ext_buffers;

    }



    if (avctx->hw_frames_ctx) {

        AVHWFramesContext    *frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;

        AVQSVFramesContext *frames_hwctx = frames_ctx->hwctx;



        if (!iopattern) {

            if (frames_hwctx->frame_type & MFX_MEMTYPE_OPAQUE_FRAME)

                iopattern = MFX_IOPATTERN_OUT_OPAQUE_MEMORY;

            else if (frames_hwctx->frame_type & MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET)

                iopattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;

        }



        frame_width  = frames_hwctx->surfaces[0].Info.Width;

        frame_height = frames_hwctx->surfaces[0].Info.Height;

    }



    if (!iopattern)

        iopattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

    q->iopattern = iopattern;



    ret = qsv_init_session(avctx, q, session, avctx->hw_frames_ctx);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error initializing an MFX session\n");

        return ret;

    }



    ret = ff_qsv_codec_id_to_mfx(avctx->codec_id);

    if (ret < 0)

        return ret;



    param.mfx.CodecId      = ret;

    param.mfx.CodecProfile = ff_qsv_profile_to_mfx(avctx->codec_id, avctx->profile);

    param.mfx.CodecLevel   = avctx->level == FF_LEVEL_UNKNOWN ? MFX_LEVEL_UNKNOWN : avctx->level;



    param.mfx.FrameInfo.BitDepthLuma   = desc->comp[0].depth;

    param.mfx.FrameInfo.BitDepthChroma = desc->comp[0].depth;

    param.mfx.FrameInfo.Shift          = desc->comp[0].depth > 8;

    param.mfx.FrameInfo.FourCC         = q->fourcc;

    param.mfx.FrameInfo.Width          = frame_width;

    param.mfx.FrameInfo.Height         = frame_height;

    param.mfx.FrameInfo.ChromaFormat   = MFX_CHROMAFORMAT_YUV420;



    switch (avctx->field_order) {

    case AV_FIELD_PROGRESSIVE:

        param.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;

        break;

    case AV_FIELD_TT:

        param.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_FIELD_TFF;

        break;

    case AV_FIELD_BB:

        param.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_FIELD_BFF;

        break;

    default:

        param.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_UNKNOWN;

        break;

    }



    param.IOPattern   = q->iopattern;

    param.AsyncDepth  = q->async_depth;

    param.ExtParam    = q->ext_buffers;

    param.NumExtParam = q->nb_ext_buffers;



    ret = MFXVideoDECODE_Init(q->session, &param);

    if (ret < 0)

        return ff_qsv_print_error(avctx, ret,

                                  "Error initializing the MFX video decoder");



    q->frame_info = param.mfx.FrameInfo;



    return 0;

}
