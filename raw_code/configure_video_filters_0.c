static int configure_video_filters(AVFilterGraph *graph, VideoState *is, const char *vfilters)

{

    static const enum PixelFormat pix_fmts[] = { PIX_FMT_YUV420P, PIX_FMT_NONE };

    char sws_flags_str[128];

    char buffersrc_args[256];

    int ret;

    AVBufferSinkParams *buffersink_params = av_buffersink_params_alloc();

    AVFilterContext *filt_src = NULL, *filt_out = NULL, *filt_format;

    AVCodecContext *codec = is->video_st->codec;



    snprintf(sws_flags_str, sizeof(sws_flags_str), "flags=%d", sws_flags);

    graph->scale_sws_opts = av_strdup(sws_flags_str);



    snprintf(buffersrc_args, sizeof(buffersrc_args),

             "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",

             codec->width, codec->height, codec->pix_fmt,

             is->video_st->time_base.num, is->video_st->time_base.den,

             codec->sample_aspect_ratio.num, codec->sample_aspect_ratio.den);



    if ((ret = avfilter_graph_create_filter(&filt_src,

                                            avfilter_get_by_name("buffer"),

                                            "ffplay_buffer", buffersrc_args, NULL,

                                            graph)) < 0)

        return ret;



    buffersink_params->pixel_fmts = pix_fmts;

    ret = avfilter_graph_create_filter(&filt_out,

                                       avfilter_get_by_name("buffersink"),

                                       "ffplay_buffersink", NULL, buffersink_params, graph);

    av_freep(&buffersink_params);

    if (ret < 0)

        return ret;



    if ((ret = avfilter_graph_create_filter(&filt_format,

                                            avfilter_get_by_name("format"),

                                            "format", "yuv420p", NULL, graph)) < 0)

        return ret;

    if ((ret = avfilter_link(filt_format, 0, filt_out, 0)) < 0)

        return ret;





    if (vfilters) {

        AVFilterInOut *outputs = avfilter_inout_alloc();

        AVFilterInOut *inputs  = avfilter_inout_alloc();



        outputs->name    = av_strdup("in");

        outputs->filter_ctx = filt_src;

        outputs->pad_idx = 0;

        outputs->next    = NULL;



        inputs->name    = av_strdup("out");

        inputs->filter_ctx = filt_format;

        inputs->pad_idx = 0;

        inputs->next    = NULL;



        if ((ret = avfilter_graph_parse(graph, vfilters, &inputs, &outputs, NULL)) < 0)

            return ret;

    } else {

        if ((ret = avfilter_link(filt_src, 0, filt_format, 0)) < 0)

            return ret;

    }



    if ((ret = avfilter_graph_config(graph, NULL)) < 0)

        return ret;



    is->in_video_filter  = filt_src;

    is->out_video_filter = filt_out;



    if (codec->codec->capabilities & CODEC_CAP_DR1) {

        is->use_dr1 = 1;

        codec->get_buffer     = codec_get_buffer;

        codec->release_buffer = codec_release_buffer;

        codec->opaque         = &is->buffer_pool;

    }



    return ret;

}
