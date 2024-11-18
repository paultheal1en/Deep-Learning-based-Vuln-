static int tta_read_header(AVFormatContext *s, AVFormatParameters *ap)
{
    TTAContext *c = s->priv_data;
    AVStream *st;
    int i, channels, bps, samplerate, datalen, framelen, start;
    start = url_ftell(&s->pb);
    if (get_le32(&s->pb) != ff_get_fourcc("TTA1"))
        return -1; // not tta file
    url_fskip(&s->pb, 2); // FIXME: flags
    channels = get_le16(&s->pb);
    bps = get_le16(&s->pb);
    samplerate = get_le32(&s->pb);
    datalen = get_le32(&s->pb);
    url_fskip(&s->pb, 4); // header crc
    framelen = 1.04489795918367346939 * samplerate;
    c->totalframes = datalen / framelen + ((datalen % framelen) ? 1 : 0);
    c->currentframe = 0;
    c->seektable = av_mallocz(sizeof(uint32_t)*c->totalframes);
    if (!c->seektable)
        return AVERROR_NOMEM;
    for (i = 0; i < c->totalframes; i++)
            c->seektable[i] = get_le32(&s->pb);
    url_fskip(&s->pb, 4); // seektable crc
    st = av_new_stream(s, 0);
//    av_set_pts_info(st, 32, 1, 1000);
    if (!st)
        return AVERROR_NOMEM;
    st->codec->codec_type = CODEC_TYPE_AUDIO;
    st->codec->codec_id = CODEC_ID_TTA;
    st->codec->channels = channels;
    st->codec->sample_rate = samplerate;
    st->codec->bits_per_sample = bps;
    st->codec->extradata_size = url_ftell(&s->pb) - start;
    if(st->codec->extradata_size+FF_INPUT_BUFFER_PADDING_SIZE <= (unsigned)st->codec->extradata_size){
        //this check is redundant as get_buffer should fail
        av_log(s, AV_LOG_ERROR, "extradata_size too large\n");
    st->codec->extradata = av_mallocz(st->codec->extradata_size+FF_INPUT_BUFFER_PADDING_SIZE);
    url_fseek(&s->pb, start, SEEK_SET); // or SEEK_CUR and -size ? :)
    get_buffer(&s->pb, st->codec->extradata, st->codec->extradata_size);
    return 0;