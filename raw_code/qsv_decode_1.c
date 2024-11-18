static int qsv_decode(AVCodecContext *avctx, QSVContext *q,

                      AVFrame *frame, int *got_frame,

                      AVPacket *avpkt)

{

    QSVFrame *out_frame;

    mfxFrameSurface1 *insurf;

    mfxFrameSurface1 *outsurf;

    mfxSyncPoint *sync;

    mfxBitstream bs = { { { 0 } } };

    int ret;



    if (avpkt->size) {

        bs.Data       = avpkt->data;

        bs.DataLength = avpkt->size;

        bs.MaxLength  = bs.DataLength;

        bs.TimeStamp  = avpkt->pts;

    }



    sync = av_mallocz(sizeof(*sync));

    if (!sync) {

        av_freep(&sync);

        return AVERROR(ENOMEM);

    }



    do {

        ret = get_surface(avctx, q, &insurf);

        if (ret < 0)

            return ret;



        ret = MFXVideoDECODE_DecodeFrameAsync(q->session, avpkt->size ? &bs : NULL,

                                              insurf, &outsurf, sync);

        if (ret == MFX_WRN_DEVICE_BUSY)

            av_usleep(500);



    } while (ret == MFX_WRN_DEVICE_BUSY || ret == MFX_ERR_MORE_SURFACE);



    if (ret != MFX_ERR_NONE &&

        ret != MFX_ERR_MORE_DATA &&

        ret != MFX_WRN_VIDEO_PARAM_CHANGED &&

        ret != MFX_ERR_MORE_SURFACE) {

        av_log(avctx, AV_LOG_ERROR, "Error during QSV decoding.\n");

        av_freep(&sync);

        return ff_qsv_error(ret);

    }



    /* make sure we do not enter an infinite loop if the SDK

     * did not consume any data and did not return anything */

    if (!*sync && !bs.DataOffset) {

        av_log(avctx, AV_LOG_WARNING, "A decode call did not consume any data\n");

        bs.DataOffset = avpkt->size;

    }



    if (*sync) {

        QSVFrame *out_frame = find_frame(q, outsurf);



        if (!out_frame) {

            av_log(avctx, AV_LOG_ERROR,

                   "The returned surface does not correspond to any frame\n");

            av_freep(&sync);

            return AVERROR_BUG;

        }



        out_frame->queued = 1;

        av_fifo_generic_write(q->async_fifo, &out_frame, sizeof(out_frame), NULL);

        av_fifo_generic_write(q->async_fifo, &sync,      sizeof(sync),      NULL);

    } else {

        av_freep(&sync);

    }



    if (!av_fifo_space(q->async_fifo) ||

        (!avpkt->size && av_fifo_size(q->async_fifo))) {

        AVFrame *src_frame;



        av_fifo_generic_read(q->async_fifo, &out_frame, sizeof(out_frame), NULL);

        av_fifo_generic_read(q->async_fifo, &sync,      sizeof(sync),      NULL);

        out_frame->queued = 0;



        do {

            ret = MFXVideoCORE_SyncOperation(q->session, *sync, 1000);

        } while (ret == MFX_WRN_IN_EXECUTION);



        av_freep(&sync);



        src_frame = out_frame->frame;



        ret = av_frame_ref(frame, src_frame);

        if (ret < 0)

            return ret;



        outsurf = out_frame->surface;



#if FF_API_PKT_PTS

FF_DISABLE_DEPRECATION_WARNINGS

        frame->pkt_pts = outsurf->Data.TimeStamp;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

        frame->pts = outsurf->Data.TimeStamp;



        frame->repeat_pict =

            outsurf->Info.PicStruct & MFX_PICSTRUCT_FRAME_TRIPLING ? 4 :

            outsurf->Info.PicStruct & MFX_PICSTRUCT_FRAME_DOUBLING ? 2 :

            outsurf->Info.PicStruct & MFX_PICSTRUCT_FIELD_REPEATED ? 1 : 0;

        frame->top_field_first =

            outsurf->Info.PicStruct & MFX_PICSTRUCT_FIELD_TFF;

        frame->interlaced_frame =

            !(outsurf->Info.PicStruct & MFX_PICSTRUCT_PROGRESSIVE);



        *got_frame = 1;

    }



    return bs.DataOffset;

}
