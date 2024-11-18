static int au_write_header(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    AVCodecContext *enc = s->streams[0]->codec;



    if (!enc->codec_tag)

        return AVERROR(EINVAL);



    ffio_wfourcc(pb, ".snd");                   /* magic number */

    avio_wb32(pb, AU_HEADER_SIZE);              /* header size */

    avio_wb32(pb, AU_UNKNOWN_SIZE);             /* data size */

    avio_wb32(pb, enc->codec_tag);              /* codec ID */

    avio_wb32(pb, enc->sample_rate);

    avio_wb32(pb, enc->channels);

    avio_wb64(pb, 0); /* annotation field */

    avio_flush(pb);



    return 0;

}
