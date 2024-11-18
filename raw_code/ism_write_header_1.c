static int ism_write_header(AVFormatContext *s)

{

    SmoothStreamingContext *c = s->priv_data;

    int ret = 0, i;

    AVOutputFormat *oformat;



    mkdir(s->filename, 0777);



    oformat = av_guess_format("ismv", NULL, NULL);

    if (!oformat) {

        ret = AVERROR_MUXER_NOT_FOUND;

        goto fail;

    }



    c->streams = av_mallocz(sizeof(*c->streams) * s->nb_streams);

    if (!c->streams) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    for (i = 0; i < s->nb_streams; i++) {

        OutputStream *os = &c->streams[i];

        AVFormatContext *ctx;

        AVStream *st;

        AVDictionary *opts = NULL;

        char buf[10];



        if (!s->streams[i]->codec->bit_rate) {

            av_log(s, AV_LOG_ERROR, "No bit rate set for stream %d\n", i);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        snprintf(os->dirname, sizeof(os->dirname), "%s/QualityLevels(%d)", s->filename, s->streams[i]->codec->bit_rate);

        mkdir(os->dirname, 0777);



        ctx = avformat_alloc_context();

        if (!ctx) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        os->ctx = ctx;

        ctx->oformat = oformat;

        ctx->interrupt_callback = s->interrupt_callback;



        if (!(st = avformat_new_stream(ctx, NULL))) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        avcodec_copy_context(st->codec, s->streams[i]->codec);

        st->sample_aspect_ratio = s->streams[i]->sample_aspect_ratio;



        ctx->pb = avio_alloc_context(os->iobuf, sizeof(os->iobuf), AVIO_FLAG_WRITE, os, NULL, ism_write, ism_seek);

        if (!ctx->pb) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        snprintf(buf, sizeof(buf), "%d", c->lookahead_count);

        av_dict_set(&opts, "ism_lookahead", buf, 0);

        av_dict_set(&opts, "movflags", "frag_custom", 0);

        if ((ret = avformat_write_header(ctx, &opts)) < 0) {

             goto fail;

        }

        os->ctx_inited = 1;

        avio_flush(ctx->pb);

        av_dict_free(&opts);

        s->streams[i]->time_base = st->time_base;

        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            c->has_video = 1;

            os->stream_type_tag = "video";

            if (st->codec->codec_id == AV_CODEC_ID_H264) {

                os->fourcc = "H264";

            } else if (st->codec->codec_id == AV_CODEC_ID_VC1) {

                os->fourcc = "WVC1";

            } else {

                av_log(s, AV_LOG_ERROR, "Unsupported video codec\n");

                ret = AVERROR(EINVAL);

                goto fail;

            }

        } else {

            c->has_audio = 1;

            os->stream_type_tag = "audio";

            if (st->codec->codec_id == AV_CODEC_ID_AAC) {

                os->fourcc = "AACL";

                os->audio_tag = 0xff;

            } else if (st->codec->codec_id == AV_CODEC_ID_WMAPRO) {

                os->fourcc = "WMAP";

                os->audio_tag = 0x0162;

            } else {

                av_log(s, AV_LOG_ERROR, "Unsupported audio codec\n");

                ret = AVERROR(EINVAL);

                goto fail;

            }

            os->packet_size = st->codec->block_align ? st->codec->block_align : 4;

        }

        get_private_data(os);

    }



    if (!c->has_video && c->min_frag_duration <= 0) {

        av_log(s, AV_LOG_WARNING, "no video stream and no min frag duration set\n");

        ret = AVERROR(EINVAL);

    }

    ret = write_manifest(s, 0);



fail:

    if (ret)

        ism_free(s);

    return ret;

}
