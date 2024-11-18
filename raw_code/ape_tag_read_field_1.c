static int ape_tag_read_field(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    uint8_t key[1024], *value;

    uint32_t size, flags;

    int i, c;



    size = avio_rl32(pb);  /* field size */

    flags = avio_rl32(pb); /* field flags */

    for (i = 0; i < sizeof(key) - 1; i++) {

        c = avio_r8(pb);

        if (c < 0x20 || c > 0x7E)

            break;

        else

            key[i] = c;

    }

    key[i] = 0;

    if (c != 0) {

        av_log(s, AV_LOG_WARNING, "Invalid APE tag key '%s'.\n", key);

        return -1;

    }

    if (size >= UINT_MAX)

        return -1;

    if (flags & APE_TAG_FLAG_IS_BINARY) {

        uint8_t filename[1024];

        AVStream *st = avformat_new_stream(s, NULL);

        if (!st)

            return AVERROR(ENOMEM);

        avio_get_str(pb, INT_MAX, filename, sizeof(filename));

        st->codec->extradata = av_malloc(size + FF_INPUT_BUFFER_PADDING_SIZE);

        if (!st->codec->extradata)

            return AVERROR(ENOMEM);

        if (avio_read(pb, st->codec->extradata, size) != size) {

            av_freep(&st->codec->extradata);

            return AVERROR(EIO);

        }

        st->codec->extradata_size = size;

        av_dict_set(&st->metadata, key, filename, 0);

        st->codec->codec_type = AVMEDIA_TYPE_ATTACHMENT;

    } else {

        value = av_malloc(size+1);

        if (!value)

            return AVERROR(ENOMEM);

        c = avio_read(pb, value, size);

        if (c < 0)

            return c;

        value[c] = 0;

        av_dict_set(&s->metadata, key, value, AV_DICT_DONT_STRDUP_VAL);

    }

    return 0;

}
