static int ast_write_header(AVFormatContext *s)

{

    ASTMuxContext *ast = s->priv_data;

    AVIOContext *pb = s->pb;

    AVCodecContext *enc;

    unsigned int codec_tag;



    if (s->nb_streams == 1) {

        enc = s->streams[0]->codec;

    } else {

        av_log(s, AV_LOG_ERROR, "only one stream is supported\n");

        return AVERROR(EINVAL);

    }



    if (enc->codec_id == AV_CODEC_ID_ADPCM_AFC) {

        av_log(s, AV_LOG_ERROR, "muxing ADPCM AFC is not implemented\n");

        return AVERROR_PATCHWELCOME;

    }



    codec_tag = ff_codec_get_tag(ff_codec_ast_tags, enc->codec_id);

    if (!codec_tag) {

        av_log(s, AV_LOG_ERROR, "unsupported codec\n");

        return AVERROR(EINVAL);

    }



    if (ast->loopstart && ast->loopend && ast->loopstart >= ast->loopend) {

        av_log(s, AV_LOG_ERROR, "loopend can't be less or equal to loopstart\n");

        return AVERROR(EINVAL);

    }



    /* Convert milliseconds to samples */

    CHECK_LOOP(start)

    CHECK_LOOP(end)



    ffio_wfourcc(pb, "STRM");



    ast->size = avio_tell(pb);

    avio_wb32(pb, 0); /* File size minus header */

    avio_wb16(pb, codec_tag);

    avio_wb16(pb, 16); /* Bit depth */

    avio_wb16(pb, enc->channels);

    avio_wb16(pb, 0xFFFF);

    avio_wb32(pb, enc->sample_rate);



    ast->samples = avio_tell(pb);

    avio_wb32(pb, 0); /* Number of samples */

    avio_wb32(pb, 0); /* Loopstart */

    avio_wb32(pb, 0); /* Loopend */

    avio_wb32(pb, 0); /* Size of first block */



    /* Unknown */

    avio_wb32(pb, 0);

    avio_wl32(pb, 0x7F);

    avio_wb64(pb, 0);

    avio_wb64(pb, 0);

    avio_wb32(pb, 0);



    avio_flush(pb);



    return 0;

}
