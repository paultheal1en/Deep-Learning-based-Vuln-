static int asf_read_single_payload(AVFormatContext *s, AVPacket *pkt,

                               ASFPacket *asf_pkt)

{

    ASFContext *asf = s->priv_data;

    AVIOContext *pb = s->pb;

    int64_t  offset;

    uint64_t size;

    unsigned char *p;

    int ret;



    if (!asf_pkt->data_size) {

        asf_pkt->data_size = asf_pkt->size_left = avio_rl32(pb); // read media object size

        if (asf_pkt->data_size <= 0)

            return AVERROR_EOF;

        if ((ret = av_new_packet(&asf_pkt->avpkt, asf_pkt->data_size)) < 0)

            return ret;

    } else

        avio_skip(pb, 4); // skip media object size

    asf_pkt->dts = avio_rl32(pb); // read presentation time

    if (asf->rep_data_len >= 8)

        avio_skip(pb, asf->rep_data_len - 8); // skip replicated data

    offset = avio_tell(pb);



    // size of the payload - size of the packet without header and padding

    if (asf->packet_size_internal)

        size = asf->packet_size_internal - offset + asf->packet_offset - asf->pad_len;

    else

        size = asf->packet_size - offset + asf->packet_offset - asf->pad_len;

    if (size > asf->packet_size) {

        av_log(s, AV_LOG_ERROR,

               "Error: invalid data packet size, offset %"PRId64".\n",

               avio_tell(pb));

        return AVERROR_INVALIDDATA;

    }

    p = asf_pkt->avpkt.data + asf_pkt->data_size - asf_pkt->size_left;

    if (size > asf_pkt->size_left || asf_pkt->size_left <= 0)

        return AVERROR_INVALIDDATA;

    if (asf_pkt->size_left > size)

        asf_pkt->size_left -= size;

    else

        asf_pkt->size_left = 0;

    if ((ret = avio_read(pb, p, size)) < 0)

        return ret;

    if (s->key && s->keylen == 20)

            ff_asfcrypt_dec(s->key, p, ret);

    if (asf->packet_size_internal)

        avio_skip(pb, asf->packet_size - asf->packet_size_internal);

    avio_skip(pb, asf->pad_len); // skip padding



    return 0;

}
