static int r3d_read_redv(AVFormatContext *s, AVPacket *pkt, Atom *atom)

{

    AVStream *st = s->streams[0];

    int tmp, tmp2;

    uint64_t pos = url_ftell(s->pb);

    unsigned dts;



    dts = get_be32(s->pb);



    tmp = get_be32(s->pb);

    dprintf(s, "frame num %d\n", tmp);



    tmp  = get_byte(s->pb); // major version

    tmp2 = get_byte(s->pb); // minor version

    dprintf(s, "version %d.%d\n", tmp, tmp2);



    tmp = get_be16(s->pb); // unknown

    dprintf(s, "unknown %d\n", tmp);



    if (tmp > 4) {

        tmp = get_be16(s->pb); // unknown

        dprintf(s, "unknown %d\n", tmp);



        tmp = get_be16(s->pb); // unknown

        dprintf(s, "unknown %d\n", tmp);



        tmp = get_be32(s->pb);

        dprintf(s, "width %d\n", tmp);

        tmp = get_be32(s->pb);

        dprintf(s, "height %d\n", tmp);



        tmp = get_be32(s->pb);

        dprintf(s, "metadata len %d\n", tmp);

    }

    tmp = atom->size - 8 - (url_ftell(s->pb) - pos);

    if (tmp < 0)

        return -1;



    if (av_get_packet(s->pb, pkt, tmp) != tmp) {

        av_log(s, AV_LOG_ERROR, "error reading video packet\n");

        return -1;

    }



    pkt->stream_index = 0;

    pkt->dts = dts;

    if (st->codec->time_base.den)

        pkt->duration = (uint64_t)st->time_base.den*

            st->codec->time_base.num/st->codec->time_base.den;

    dprintf(s, "pkt dts %lld duration %d\n", pkt->dts, pkt->duration);



    return 0;

}
