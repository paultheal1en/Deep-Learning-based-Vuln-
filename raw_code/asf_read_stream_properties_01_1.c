static int asf_read_stream_properties(AVFormatContext *s, const GUIDParseTable *g)

{

    ASFContext *asf = s->priv_data;

    AVIOContext *pb = s->pb;

    uint64_t size;

    uint32_t err_data_len, ts_data_len; // type specific data length

    uint16_t flags;

    ff_asf_guid stream_type;

    enum AVMediaType type;

    int i, ret;

    uint8_t stream_index;

    AVStream *st;

    ASFStream *asf_st;



    // ASF file must not contain more than 128 streams according to the specification

    if (asf->nb_streams >= ASF_MAX_STREAMS)

        return AVERROR_INVALIDDATA;



    size = avio_rl64(pb);

    ff_get_guid(pb, &stream_type);

    if (!ff_guidcmp(&stream_type, &ff_asf_audio_stream))

        type = AVMEDIA_TYPE_AUDIO;

    else if (!ff_guidcmp(&stream_type, &ff_asf_video_stream))

        type = AVMEDIA_TYPE_VIDEO;

    else if (!ff_guidcmp(&stream_type, &ff_asf_jfif_media))

        type = AVMEDIA_TYPE_VIDEO;

    else if (!ff_guidcmp(&stream_type, &ff_asf_command_stream))

        type = AVMEDIA_TYPE_DATA;

    else if (!ff_guidcmp(&stream_type,

                         &ff_asf_ext_stream_embed_stream_header))

        type = AVMEDIA_TYPE_UNKNOWN;

    else

        return AVERROR_INVALIDDATA;



    ff_get_guid(pb, &stream_type); // error correction type

    avio_skip(pb, 8); // skip the time offset

    ts_data_len      = avio_rl32(pb);

    err_data_len     = avio_rl32(pb);

    flags            = avio_rl16(pb); // bit 15 - Encrypted Content



    stream_index = flags & ASF_STREAM_NUM;

    for (i = 0; i < asf->nb_streams; i++)

        if (stream_index == asf->asf_st[i]->stream_index) {

            av_log(s, AV_LOG_WARNING,

                   "Duplicate stream found, this stream will be ignored.\n");

            align_position(pb, asf->offset, size);

            return 0;

        }



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);

    avpriv_set_pts_info(st, 32, 1, 1000); // pts should be dword, in milliseconds

    st->codec->codec_type = type;

    asf->asf_st[asf->nb_streams] = av_mallocz(sizeof(*asf_st));

    if (!asf->asf_st[asf->nb_streams])

        return AVERROR(ENOMEM);

    asf_st                       = asf->asf_st[asf->nb_streams];

    asf_st->stream_index         = stream_index;

    asf_st->index                = st->index;

    asf_st->indexed              = 0;

    st->id                       = flags & ASF_STREAM_NUM;

    av_init_packet(&asf_st->pkt.avpkt);

    asf_st->pkt.data_size        = 0;

    avio_skip(pb, 4); // skip reserved field



    switch (type) {

    case AVMEDIA_TYPE_AUDIO:

        asf_st->type = AVMEDIA_TYPE_AUDIO;

        if ((ret = ff_get_wav_header(s, pb, st->codec, ts_data_len)) < 0)

            return ret;

        break;

    case AVMEDIA_TYPE_VIDEO:

        asf_st->type = AVMEDIA_TYPE_VIDEO;

        if ((ret = parse_video_info(pb, st)) < 0)

            return ret;

        break;

    default:

        avio_skip(pb, ts_data_len);

        break;

    }



    if (err_data_len) {

        if (type == AVMEDIA_TYPE_AUDIO) {

            uint8_t span = avio_r8(pb);

            if (span > 1) {

                asf_st->span              = span;

                asf_st->virtual_pkt_len   = avio_rl16(pb);

                asf_st->virtual_chunk_len = avio_rl16(pb);

                if (!asf_st->virtual_chunk_len || !asf_st->virtual_pkt_len)

                    return AVERROR_INVALIDDATA;

                avio_skip(pb, err_data_len - 5);

            } else

                avio_skip(pb, err_data_len - 1);

        } else

            avio_skip(pb, err_data_len);

    }



    asf->nb_streams++;

    align_position(pb, asf->offset, size);



    return 0;

}
