static int parse_picture(AVFormatContext *s, uint8_t *buf, int buf_size)

{

    const CodecMime *mime = ff_id3v2_mime_tags;

    enum  CodecID      id = CODEC_ID_NONE;

    uint8_t mimetype[64], *desc = NULL, *data = NULL;

    AVIOContext *pb = NULL;

    AVStream *st;

    int type, width, height;

    int len, ret = 0;



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    pb = avio_alloc_context(buf, buf_size, 0, NULL, NULL, NULL, NULL);

    if (!pb)

        return AVERROR(ENOMEM);



    /* read the picture type */

    type      = avio_rb32(pb);

    if (type >= FF_ARRAY_ELEMS(ff_id3v2_picture_types) || type < 0) {

        av_log(s, AV_LOG_ERROR, "Invalid picture type: %d.\n", type);

        ret = AVERROR_INVALIDDATA;

        goto fail;

    }



    /* picture mimetype */

    len  = avio_rb32(pb);

    if (len <= 0 ||

        avio_read(pb, mimetype, FFMIN(len, sizeof(mimetype) - 1)) != len) {

        av_log(s, AV_LOG_ERROR, "Could not read mimetype from an attached "

               "picture.\n");

        ret = AVERROR_INVALIDDATA;

        goto fail;

    }

    mimetype[len] = 0;



    while (mime->id != CODEC_ID_NONE) {

        if (!strncmp(mime->str, mimetype, sizeof(mimetype))) {

            id = mime->id;

            break;

        }

        mime++;

    }

    if (id == CODEC_ID_NONE) {

        av_log(s, AV_LOG_ERROR, "Unknown attached picture mimetype: %s.\n",

               mimetype);

        ret = AVERROR_INVALIDDATA;

        goto fail;

    }



    /* picture description */

    len = avio_rb32(pb);

    if (len > 0) {

        if (!(desc = av_malloc(len + 1))) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        if (avio_read(pb, desc, len) != len) {

            ret = AVERROR(EIO);

            goto fail;

        }

        desc[len] = 0;

    }



    /* picture metadata */

    width  = avio_rb32(pb);

    height = avio_rb32(pb);

    avio_skip(pb, 8);



    /* picture data */

    len = avio_rb32(pb);

    if (len <= 0) {

        ret = AVERROR_INVALIDDATA;

        goto fail;

    }

    if (!(data = av_malloc(len))) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }

    if (avio_read(pb, data, len) != len) {

        ret = AVERROR(EIO);

        goto fail;

    }



    av_init_packet(&st->attached_pic);

    st->attached_pic.data         = data;

    st->attached_pic.size         = len;

    st->attached_pic.destruct     = av_destruct_packet;

    st->attached_pic.stream_index = st->index;

    st->attached_pic.flags       |= AV_PKT_FLAG_KEY;



    st->disposition      |= AV_DISPOSITION_ATTACHED_PIC;

    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id   = id;

    st->codec->width      = width;

    st->codec->height     = height;

    av_dict_set(&st->metadata, "comment", ff_id3v2_picture_types[type], 0);

    if (desc)

        av_dict_set(&st->metadata, "title",   desc, AV_DICT_DONT_STRDUP_VAL);



    av_freep(&pb);



    return 0;



fail:

    av_freep(&desc);

    av_freep(&data);

    av_freep(&pb);

    return ret;



}
