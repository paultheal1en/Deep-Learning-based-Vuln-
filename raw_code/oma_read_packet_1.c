static int oma_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    OMAContext *oc  = s->priv_data;

    AVStream *st    = s->streams[0];

    int packet_size = st->codec->block_align;

    int byte_rate   = st->codec->bit_rate >> 3;

    int64_t pos     = avio_tell(s->pb);

    int ret         = av_get_packet(s->pb, pkt, packet_size);



    if (ret < packet_size)

        pkt->flags |= AV_PKT_FLAG_CORRUPT;



    if (ret < 0)

        return ret;

    if (!ret)

        return AVERROR_EOF;



    pkt->stream_index = 0;



    if (pos > 0) {

        pkt->pts =

        pkt->dts = av_rescale(pos, st->time_base.den,

                              byte_rate * (int64_t)st->time_base.num);

    }



    if (oc->encrypted) {

        /* previous unencrypted block saved in IV for

         * the next packet (CBC mode) */

        if (ret == packet_size)

            av_des_crypt(&oc->av_des, pkt->data, pkt->data,

                         (packet_size >> 3), oc->iv, 1);

        else

            memset(oc->iv, 0, 8);

    }



    return ret;

}
