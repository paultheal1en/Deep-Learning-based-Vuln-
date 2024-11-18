static int tta_read_header(AVFormatContext *s)

{

    TTAContext *c = s->priv_data;

    AVStream *st;

    int i, channels, bps, samplerate;

    uint64_t framepos, start_offset;

    uint32_t nb_samples, crc;



    ff_id3v1_read(s);



    start_offset = avio_tell(s->pb);

    ffio_init_checksum(s->pb, tta_check_crc, UINT32_MAX);

    if (avio_rl32(s->pb) != AV_RL32("TTA1"))

        return AVERROR_INVALIDDATA;



    avio_skip(s->pb, 2); // FIXME: flags

    channels = avio_rl16(s->pb);

    bps = avio_rl16(s->pb);

    samplerate = avio_rl32(s->pb);

    if(samplerate <= 0 || samplerate > 1000000){

        av_log(s, AV_LOG_ERROR, "nonsense samplerate\n");

        return AVERROR_INVALIDDATA;

    }



    nb_samples = avio_rl32(s->pb);

    if (!nb_samples) {

        av_log(s, AV_LOG_ERROR, "invalid number of samples\n");

        return AVERROR_INVALIDDATA;

    }



    crc = ffio_get_checksum(s->pb) ^ UINT32_MAX;

    if (crc != avio_rl32(s->pb)) {

        av_log(s, AV_LOG_ERROR, "Header CRC error\n");

        return AVERROR_INVALIDDATA;

    }



    c->frame_size      = samplerate * 256 / 245;

    c->last_frame_size = nb_samples % c->frame_size;

    if (!c->last_frame_size)

        c->last_frame_size = c->frame_size;

    c->totalframes = nb_samples / c->frame_size + (c->last_frame_size < c->frame_size);

    c->currentframe = 0;



    if(c->totalframes >= UINT_MAX/sizeof(uint32_t) || c->totalframes <= 0){

        av_log(s, AV_LOG_ERROR, "totalframes %d invalid\n", c->totalframes);

        return AVERROR_INVALIDDATA;

    }



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    avpriv_set_pts_info(st, 64, 1, samplerate);

    st->start_time = 0;

    st->duration = nb_samples;



    framepos = avio_tell(s->pb) + 4*c->totalframes + 4;



    if (ff_alloc_extradata(st->codec, avio_tell(s->pb) - start_offset))

        return AVERROR(ENOMEM);



    avio_seek(s->pb, start_offset, SEEK_SET);

    avio_read(s->pb, st->codec->extradata, st->codec->extradata_size);



    ffio_init_checksum(s->pb, tta_check_crc, UINT32_MAX);

    for (i = 0; i < c->totalframes; i++) {

        uint32_t size = avio_rl32(s->pb);

        av_add_index_entry(st, framepos, i * c->frame_size, size, 0,

                           AVINDEX_KEYFRAME);

        framepos += size;

    }

    crc = ffio_get_checksum(s->pb) ^ UINT32_MAX;

    if (crc != avio_rl32(s->pb)) {

        av_log(s, AV_LOG_ERROR, "Seek table CRC error\n");

        return AVERROR_INVALIDDATA;

    }



    st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

    st->codec->codec_id = AV_CODEC_ID_TTA;

    st->codec->channels = channels;

    st->codec->sample_rate = samplerate;

    st->codec->bits_per_coded_sample = bps;



    if (s->pb->seekable) {

        int64_t pos = avio_tell(s->pb);

        ff_ape_parse_tag(s);

        avio_seek(s->pb, pos, SEEK_SET);

    }



    return 0;

}
