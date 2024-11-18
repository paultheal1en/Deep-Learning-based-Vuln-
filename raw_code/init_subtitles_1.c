static av_cold int init_subtitles(AVFilterContext *ctx, const char *args)

{

    int ret, sid;

    AVFormatContext *fmt = NULL;

    AVCodecContext *dec_ctx = NULL;

    AVCodec *dec = NULL;

    AVStream *st;

    AVPacket pkt;

    AssContext *ass = ctx->priv;



    /* Init libass */

    ret = init(ctx, args, &subtitles_class);

    if (ret < 0)

        return ret;

    ass->track = ass_new_track(ass->library);

    if (!ass->track) {

        av_log(ctx, AV_LOG_ERROR, "Could not create a libass track\n");

        return AVERROR(EINVAL);

    }



    /* Open subtitles file */

    ret = avformat_open_input(&fmt, ass->filename, NULL, NULL);

    if (ret < 0) {

        av_log(ctx, AV_LOG_ERROR, "Unable to open %s\n", ass->filename);

        goto end;

    }

    ret = avformat_find_stream_info(fmt, NULL);

    if (ret < 0)

        goto end;



    /* Locate subtitles stream */

    ret = av_find_best_stream(fmt, AVMEDIA_TYPE_SUBTITLE, -1, -1, NULL, 0);

    if (ret < 0) {

        av_log(ctx, AV_LOG_ERROR, "Unable to locate subtitle stream in %s\n",

               ass->filename);

        goto end;

    }

    sid = ret;

    st = fmt->streams[sid];



    /* Open decoder */

    dec_ctx = st->codec;

    dec = avcodec_find_decoder(dec_ctx->codec_id);

    if (!dec) {

        av_log(ctx, AV_LOG_ERROR, "Failed to find subtitle codec %s\n",

               avcodec_get_name(dec_ctx->codec_id));

        return AVERROR(EINVAL);

    }

    ret = avcodec_open2(dec_ctx, dec, NULL);

    if (ret < 0)

        goto end;



    /* Decode subtitles and push them into the renderer (libass) */

    if (dec_ctx->subtitle_header)

        ass_process_codec_private(ass->track,

                                  dec_ctx->subtitle_header,

                                  dec_ctx->subtitle_header_size);

    av_init_packet(&pkt);

    pkt.data = NULL;

    pkt.size = 0;

    while (av_read_frame(fmt, &pkt) >= 0) {

        int i, got_subtitle;

        AVSubtitle sub;



        if (pkt.stream_index == sid) {

            ret = avcodec_decode_subtitle2(dec_ctx, &sub, &got_subtitle, &pkt);

            if (ret < 0 || !got_subtitle)

                break;

            for (i = 0; i < sub.num_rects; i++) {

                char *ass_line = sub.rects[i]->ass;

                if (!ass_line)

                    break;

                ass_process_data(ass->track, ass_line, strlen(ass_line));

            }

        }

        av_free_packet(&pkt);

        avsubtitle_free(&sub);

    }



end:

    if (fmt)

        avformat_close_input(&fmt);

    if (dec_ctx)

        avcodec_close(dec_ctx);

    return ret;

}
