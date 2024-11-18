static int ipvideo_decode_frame(AVCodecContext *avctx,

                                void *data, int *got_frame,

                                AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    IpvideoContext *s = avctx->priv_data;

    AVFrame *frame = data;

    int ret;

    int send_buffer;

    int frame_format;

    int video_data_size;



    if (av_packet_get_side_data(avpkt, AV_PKT_DATA_PARAM_CHANGE, NULL)) {

        av_frame_unref(s->last_frame);

        av_frame_unref(s->second_last_frame);

    }



    if (buf_size < 6)

        return AVERROR_INVALIDDATA;



    frame_format         = AV_RL8(buf);

    send_buffer          = AV_RL8(buf + 1);

    video_data_size      = AV_RL16(buf + 2);

    s->decoding_map_size = AV_RL16(buf + 4);



    if (frame_format != 0x11)

        av_log(avctx, AV_LOG_ERROR, "Frame type 0x%02X unsupported\n", frame_format);



    if (! s->decoding_map_size) {

        av_log(avctx, AV_LOG_ERROR, "Empty decoding map\n");

        return AVERROR_INVALIDDATA;

    }



    bytestream2_init(&s->stream_ptr, buf + 6, video_data_size);



    /* decoding map contains 4 bits of information per 8x8 block */

    s->decoding_map = buf + 6 + video_data_size;



    /* ensure we can't overread the packet */

    if (buf_size < 6 + s->decoding_map_size + video_data_size) {

        av_log(avctx, AV_LOG_ERROR, "Invalid IP packet size\n");

        return AVERROR_INVALIDDATA;

    }



    if ((ret = ff_get_buffer(avctx, frame, AV_GET_BUFFER_FLAG_REF)) < 0)

        return ret;



    if (!s->is_16bpp) {

        int size;

        const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE, &size);

        if (pal && size == AVPALETTE_SIZE) {

            frame->palette_has_changed = 1;

            memcpy(s->pal, pal, AVPALETTE_SIZE);

        } else if (pal) {

            av_log(avctx, AV_LOG_ERROR, "Palette size %d is wrong\n", size);

        }

    }



    ipvideo_decode_opcodes(s, frame);



    *got_frame = send_buffer;



    /* shuffle frames */

    av_frame_unref(s->second_last_frame);

    FFSWAP(AVFrame*, s->second_last_frame, s->last_frame);

    if ((ret = av_frame_ref(s->last_frame, frame)) < 0)

        return ret;



    /* report that the buffer was completely consumed */

    return buf_size;

}
