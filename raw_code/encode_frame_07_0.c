static int encode_frame(AVCodecContext *avctx, QSVEncContext *q,

                        const AVFrame *frame)

{

    AVPacket new_pkt = { 0 };

    mfxBitstream *bs;



    mfxFrameSurface1 *surf = NULL;

    mfxSyncPoint sync      = NULL;

    int ret;



    if (frame) {

        ret = submit_frame(q, frame, &surf);

        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "Error submitting the frame for encoding.\n");

            return ret;

        }

    }



    ret = av_new_packet(&new_pkt, q->packet_size);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error allocating the output packet\n");

        return ret;

    }



    bs = av_mallocz(sizeof(*bs));

    if (!bs) {

        av_packet_unref(&new_pkt);

        return AVERROR(ENOMEM);

    }

    bs->Data      = new_pkt.data;

    bs->MaxLength = new_pkt.size;



    do {

        ret = MFXVideoENCODE_EncodeFrameAsync(q->session, NULL, surf, bs, &sync);

        if (ret == MFX_WRN_DEVICE_BUSY)

            av_usleep(1);

    } while (ret > 0);



    if (ret < 0) {

        av_packet_unref(&new_pkt);

        av_freep(&bs);

        return (ret == MFX_ERR_MORE_DATA) ? 0 : ff_qsv_error(ret);

    }



    if (ret == MFX_WRN_INCOMPATIBLE_VIDEO_PARAM && frame->interlaced_frame)

        print_interlace_msg(avctx, q);



    if (sync) {

        av_fifo_generic_write(q->async_fifo, &new_pkt, sizeof(new_pkt), NULL);

        av_fifo_generic_write(q->async_fifo, &sync,    sizeof(sync),    NULL);

        av_fifo_generic_write(q->async_fifo, &bs,      sizeof(bs),    NULL);

    } else {

        av_packet_unref(&new_pkt);

        av_freep(&bs);

    }



    return 0;

}
