static int nut_write_header(AVFormatContext * avf) {

    NUTContext * priv = avf->priv_data;

    AVIOContext * bc = avf->pb;

    nut_muxer_opts_tt mopts = {

        .output = {

            .priv = bc,

            .write = av_write,

        },

        .alloc = { av_malloc, av_realloc, av_free },

        .write_index = 1,

        .realtime_stream = 0,

        .max_distance = 32768,

        .fti = NULL,

    };

    nut_stream_header_tt * s;

    int i;



    priv->s = s = av_mallocz((avf->nb_streams + 1) * sizeof*s);





    for (i = 0; i < avf->nb_streams; i++) {

        AVCodecContext * codec = avf->streams[i]->codec;

        int j;

        int fourcc = 0;

        int num, denom, ssize;



        s[i].type = codec->codec_type == AVMEDIA_TYPE_VIDEO ? NUT_VIDEO_CLASS : NUT_AUDIO_CLASS;



        if (codec->codec_tag) fourcc = codec->codec_tag;

        else fourcc = ff_codec_get_tag(nut_tags, codec->codec_id);



        if (!fourcc) {

            if (codec->codec_type == AVMEDIA_TYPE_VIDEO) fourcc = ff_codec_get_tag(ff_codec_bmp_tags, codec->codec_id);

            if (codec->codec_type == AVMEDIA_TYPE_AUDIO) fourcc = ff_codec_get_tag(ff_codec_wav_tags, codec->codec_id);

        }



        s[i].fourcc_len = 4;

        s[i].fourcc = av_malloc(s[i].fourcc_len);

        for (j = 0; j < s[i].fourcc_len; j++) s[i].fourcc[j] = (fourcc >> (j*8)) & 0xFF;



        ff_parse_specific_params(codec, &num, &ssize, &denom);

        avpriv_set_pts_info(avf->streams[i], 60, denom, num);



        s[i].time_base.num = denom;

        s[i].time_base.den = num;



        s[i].fixed_fps = 0;

        s[i].decode_delay = codec->has_b_frames;

        s[i].codec_specific_len = codec->extradata_size;

        s[i].codec_specific = codec->extradata;



        if (codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            s[i].width = codec->width;

            s[i].height = codec->height;

            s[i].sample_width = 0;

            s[i].sample_height = 0;

            s[i].colorspace_type = 0;

        } else {

            s[i].samplerate_num = codec->sample_rate;

            s[i].samplerate_denom = 1;

            s[i].channel_count = codec->channels;

        }

    }



    s[avf->nb_streams].type = -1;

    priv->nut = nut_muxer_init(&mopts, s, NULL);



    return 0;

}