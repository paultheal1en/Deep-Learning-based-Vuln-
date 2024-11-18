static void get_attachment(AVFormatContext *s, AVIOContext *pb, int length)

{

    char mime[1024];

    char description[1024];

    unsigned int filesize;

    AVStream *st;

    int64_t pos = avio_tell(pb);



    avio_get_str16le(pb, INT_MAX, mime, sizeof(mime));

    if (strcmp(mime, "image/jpeg"))

        goto done;



    avio_r8(pb);

    avio_get_str16le(pb, INT_MAX, description, sizeof(description));

    filesize = avio_rl32(pb);

    if (!filesize)

        goto done;



    st = avformat_new_stream(s, NULL);

    if (!st)

        goto done;

    av_dict_set(&st->metadata, "title", description, 0);

    st->codec->codec_id   = AV_CODEC_ID_MJPEG;

    st->codec->codec_type = AVMEDIA_TYPE_ATTACHMENT;

    st->codec->extradata  = av_mallocz(filesize);


    if (!st->codec->extradata)

        goto done;

    st->codec->extradata_size = filesize;

    avio_read(pb, st->codec->extradata, filesize);

done:

    avio_seek(pb, pos + length, SEEK_SET);

}