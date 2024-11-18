static int w64_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    int64_t size;

    AVIOContext *pb  = s->pb;

    WAVContext    *wav = s->priv_data;

    AVStream *st;

    uint8_t guid[16];



    avio_read(pb, guid, 16);

    if (memcmp(guid, guid_riff, 16))

        return -1;



    if (avio_rl64(pb) < 16 + 8 + 16 + 8 + 16 + 8) /* riff + wave + fmt + sizes */

        return -1;



    avio_read(pb, guid, 16);

    if (memcmp(guid, guid_wave, 16)) {

        av_log(s, AV_LOG_ERROR, "could not find wave guid\n");

        return -1;

    }



    size = find_guid(pb, guid_fmt);

    if (size < 0) {

        av_log(s, AV_LOG_ERROR, "could not find fmt guid\n");

        return -1;

    }



    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR(ENOMEM);



    /* subtract chunk header size - normal wav file doesn't count it */

    ff_get_wav_header(pb, st->codec, size - 24);

    avio_skip(pb, FFALIGN(size, INT64_C(8)) - size);



    st->need_parsing = AVSTREAM_PARSE_FULL;



    av_set_pts_info(st, 64, 1, st->codec->sample_rate);



    size = find_guid(pb, guid_data);

    if (size < 0) {

        av_log(s, AV_LOG_ERROR, "could not find data guid\n");

        return -1;

    }

    wav->data_end = avio_tell(pb) + size - 24;

    wav->w64      = 1;



    return 0;

}
