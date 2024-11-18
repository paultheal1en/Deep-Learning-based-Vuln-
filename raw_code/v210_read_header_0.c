static int v210_read_header(AVFormatContext *ctx)

{

    V210DemuxerContext *s = ctx->priv_data;

    AVStream *st;



    st = avformat_new_stream(ctx, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;



    st->codec->codec_id = ctx->iformat->raw_codec_id;



    avpriv_set_pts_info(st, 64, s->framerate.den, s->framerate.num);



    st->codec->width    = s->width;

    st->codec->height   = s->height;

    st->codec->pix_fmt  = ctx->iformat->raw_codec_id == AV_CODEC_ID_V210 ?

                          AV_PIX_FMT_YUV422P10 : AV_PIX_FMT_YUV422P16;

    st->codec->bit_rate = av_rescale_q(GET_PACKET_SIZE(s->width, s->height),

                                       (AVRational){8,1}, st->time_base);



    return 0;

}
