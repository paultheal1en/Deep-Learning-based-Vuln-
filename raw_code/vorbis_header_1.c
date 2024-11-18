vorbis_header (AVFormatContext * s, int idx)

{

    ogg_t *ogg = s->priv_data;

    ogg_stream_t *os = ogg->streams + idx;

    AVStream *st = s->streams[idx];

    oggvorbis_private_t *priv;



    if (os->seq > 2)

        return 0;



    if (os->seq == 0) {

        os->private = av_mallocz(sizeof(oggvorbis_private_t));

        if (!os->private)

            return 0;

    }



    priv = os->private;

    priv->len[os->seq] = os->psize;

    priv->packet[os->seq] = av_mallocz(os->psize);

    memcpy(priv->packet[os->seq], os->buf + os->pstart, os->psize);

    if (os->buf[os->pstart] == 1) {

        uint8_t *p = os->buf + os->pstart + 11; //skip up to the audio channels

        st->codec->channels = *p++;

        st->codec->sample_rate = AV_RL32(p);

        p += 8; //skip maximum and and nominal bitrate

        st->codec->bit_rate = AV_RL32(p); //Minimum bitrate



        st->codec->codec_type = CODEC_TYPE_AUDIO;

        st->codec->codec_id = CODEC_ID_VORBIS;



        st->time_base.num = 1;

        st->time_base.den = st->codec->sample_rate;

    } else if (os->buf[os->pstart] == 3) {

        vorbis_comment (s, os->buf + os->pstart + 7, os->psize - 8);

    } else {

        st->codec->extradata_size =

            fixup_vorbis_headers(s, priv, &st->codec->extradata);

    }



    return os->seq < 3;

}
