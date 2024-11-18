static av_cold int vc2_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                                      const AVFrame *frame, int *got_packet)

{

    int ret = 0;

    int sig_size = 256;

    VC2EncContext *s = avctx->priv_data;

    const char aux_data[] = LIBAVCODEC_IDENT;

    const int aux_data_size = sizeof(aux_data);

    const int header_size = 100 + aux_data_size;

    int64_t max_frame_bytes, r_bitrate = avctx->bit_rate >> (s->interlaced);



    s->avctx = avctx;

    s->size_scaler = 1;

    s->prefix_bytes = 0;

    s->last_parse_code = 0;

    s->next_parse_offset = 0;



    /* Rate control */

    max_frame_bytes = (av_rescale(r_bitrate, s->avctx->time_base.num,

                                  s->avctx->time_base.den) >> 3) - header_size;



    /* Find an appropriate size scaler */

    while (sig_size > 255) {

        s->slice_max_bytes = FFALIGN(av_rescale(max_frame_bytes, 1,

                                                s->num_x*s->num_y), s->size_scaler);

        s->slice_max_bytes += 4 + s->prefix_bytes;

        sig_size = s->slice_max_bytes/s->size_scaler; /* Signalled slize size */

        s->size_scaler <<= 1;

    }



    s->slice_min_bytes = s->slice_max_bytes - s->slice_max_bytes*(s->tolerance/100.0f);



    ret = encode_frame(s, avpkt, frame, aux_data, header_size, s->interlaced);

    if (ret)

        return ret;

    if (s->interlaced) {

        ret = encode_frame(s, avpkt, frame, aux_data, header_size, 2);

        if (ret)

            return ret;

    }



    flush_put_bits(&s->pb);

    avpkt->size = put_bits_count(&s->pb) >> 3;



    *got_packet = 1;



    return 0;

}
