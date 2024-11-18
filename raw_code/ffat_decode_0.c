static int ffat_decode(AVCodecContext *avctx, void *data,

                       int *got_frame_ptr, AVPacket *avpkt)

{

    ATDecodeContext *at = avctx->priv_data;

    AVFrame *frame = data;

    int pkt_size = avpkt->size;

    AVPacket filtered_packet = {0};

    OSStatus ret;

    AudioBufferList out_buffers;



    if (avctx->codec_id == AV_CODEC_ID_AAC && avpkt->size > 2 &&

        (AV_RB16(avpkt->data) & 0xfff0) == 0xfff0) {

        AVPacket filter_pkt = {0};

        if (!at->bsf) {

            const AVBitStreamFilter *bsf = av_bsf_get_by_name("aac_adtstoasc");

            if(!bsf)

                return AVERROR_BSF_NOT_FOUND;

            if ((ret = av_bsf_alloc(bsf, &at->bsf)))

                return ret;

            if (((ret = avcodec_parameters_from_context(at->bsf->par_in, avctx)) < 0) ||

                ((ret = av_bsf_init(at->bsf)) < 0)) {

                av_bsf_free(&at->bsf);

                return ret;

            }

        }



        if ((ret = av_packet_ref(&filter_pkt, avpkt)) < 0)

            return ret;



        if ((ret = av_bsf_send_packet(at->bsf, &filter_pkt)) < 0) {

            av_packet_unref(&filter_pkt);

            return ret;

        }



        if ((ret = av_bsf_receive_packet(at->bsf, &filtered_packet)) < 0)

            return ret;



        at->extradata = at->bsf->par_out->extradata;

        at->extradata_size = at->bsf->par_out->extradata_size;



        avpkt = &filtered_packet;

    }



    if (!at->converter) {

        if ((ret = ffat_create_decoder(avctx, avpkt)) < 0) {

            av_packet_unref(&filtered_packet);

            return ret;

        }

    }



    out_buffers = (AudioBufferList){

        .mNumberBuffers = 1,

        .mBuffers = {

            {

                .mNumberChannels = avctx->channels,

                .mDataByteSize = av_get_bytes_per_sample(avctx->sample_fmt) * avctx->frame_size

                                 * avctx->channels,

            }

        }

    };



    av_packet_unref(&at->new_in_pkt);



    if (avpkt->size) {

        if (filtered_packet.data) {

            at->new_in_pkt = filtered_packet;

        } else if ((ret = av_packet_ref(&at->new_in_pkt, avpkt)) < 0) {

            return ret;

        }

    } else {

        at->eof = 1;

    }



    frame->sample_rate = avctx->sample_rate;



    frame->nb_samples = avctx->frame_size;



    out_buffers.mBuffers[0].mData = at->decoded_data;



    ret = AudioConverterFillComplexBuffer(at->converter, ffat_decode_callback, avctx,

                                          &frame->nb_samples, &out_buffers, NULL);

    if ((!ret || ret == 1) && frame->nb_samples) {

        if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)

            return ret;

        ffat_copy_samples(avctx, frame);

        *got_frame_ptr = 1;

        if (at->last_pts != AV_NOPTS_VALUE) {

            frame->pts = at->last_pts;

#if FF_API_PKT_PTS

FF_DISABLE_DEPRECATION_WARNINGS

            frame->pkt_pts = at->last_pts;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

            at->last_pts = avpkt->pts;

        }

    } else if (ret && ret != 1) {

        av_log(avctx, AV_LOG_WARNING, "Decode error: %i\n", ret);

    } else {

        at->last_pts = avpkt->pts;

    }



    return pkt_size;

}
