static int put_flac_codecpriv(AVFormatContext *s,

                              AVIOContext *pb, AVCodecContext *codec)

{

    int write_comment = (codec->channel_layout &&

                         !(codec->channel_layout & ~0x3ffffULL) &&

                         !ff_flac_is_native_layout(codec->channel_layout));

    int ret = ff_flac_write_header(pb, codec->extradata, codec->extradata_size,

                                   !write_comment);



    if (ret < 0)

        return ret;



    if (write_comment) {

        const char *vendor = (s->flags & AVFMT_FLAG_BITEXACT) ?

                             "Lavf" : LIBAVFORMAT_IDENT;

        AVDictionary *dict = NULL;

        uint8_t buf[32], *data, *p;

        int len;



        snprintf(buf, sizeof(buf), "0x%"PRIx64, codec->channel_layout);

        av_dict_set(&dict, "WAVEFORMATEXTENSIBLE_CHANNEL_MASK", buf, 0);



        len = ff_vorbiscomment_length(dict, vendor);

        data = av_malloc(len + 4);

        if (!data) {

            av_dict_free(&dict);

            return AVERROR(ENOMEM);

        }



        data[0] = 0x84;

        AV_WB24(data + 1, len);



        p = data + 4;

        ff_vorbiscomment_write(&p, &dict, vendor);



        avio_write(pb, data, len + 4);



        av_freep(&data);

        av_dict_free(&dict);

    }



    return 0;

}
