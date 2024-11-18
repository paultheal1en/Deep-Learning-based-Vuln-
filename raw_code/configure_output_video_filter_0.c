static int configure_output_video_filter(FilterGraph *fg, OutputFilter *ofilter, AVFilterInOut *out)

{

    char *pix_fmts;

    OutputStream *ost = ofilter->ost;

    OutputFile    *of = output_files[ost->file_index];

    AVFilterContext *last_filter = out->filter_ctx;

    int pad_idx = out->pad_idx;

    int ret;

    char name[255];



    snprintf(name, sizeof(name), "output stream %d:%d", ost->file_index, ost->index);

    ret = avfilter_graph_create_filter(&ofilter->filter,

                                       avfilter_get_by_name("buffersink"),

                                       name, NULL, NULL, fg->graph);

    if (ret < 0)

        return ret;



    if (!hw_device_ctx && (ofilter->width || ofilter->height)) {

        char args[255];

        AVFilterContext *filter;



        snprintf(args, sizeof(args), "%d:%d:0x%X",

                 ofilter->width, ofilter->height,

                 (unsigned)ost->sws_flags);

        snprintf(name, sizeof(name), "scaler for output stream %d:%d",

                 ost->file_index, ost->index);

        if ((ret = avfilter_graph_create_filter(&filter, avfilter_get_by_name("scale"),

                                                name, args, NULL, fg->graph)) < 0)

            return ret;

        if ((ret = avfilter_link(last_filter, pad_idx, filter, 0)) < 0)

            return ret;



        last_filter = filter;

        pad_idx = 0;

    }



    if ((pix_fmts = choose_pix_fmts(ofilter))) {

        AVFilterContext *filter;

        snprintf(name, sizeof(name), "pixel format for output stream %d:%d",

                 ost->file_index, ost->index);

        ret = avfilter_graph_create_filter(&filter,

                                           avfilter_get_by_name("format"),

                                           "format", pix_fmts, NULL, fg->graph);

        av_freep(&pix_fmts);

        if (ret < 0)

            return ret;

        if ((ret = avfilter_link(last_filter, pad_idx, filter, 0)) < 0)

            return ret;



        last_filter = filter;

        pad_idx     = 0;

    }



    if (ost->frame_rate.num) {

        AVFilterContext *fps;

        char args[255];



        snprintf(args, sizeof(args), "fps=%d/%d", ost->frame_rate.num,

                 ost->frame_rate.den);

        snprintf(name, sizeof(name), "fps for output stream %d:%d",

                 ost->file_index, ost->index);

        ret = avfilter_graph_create_filter(&fps, avfilter_get_by_name("fps"),

                                           name, args, NULL, fg->graph);

        if (ret < 0)

            return ret;



        ret = avfilter_link(last_filter, pad_idx, fps, 0);

        if (ret < 0)

            return ret;

        last_filter = fps;

        pad_idx = 0;

    }



    snprintf(name, sizeof(name), "trim for output stream %d:%d",

             ost->file_index, ost->index);

    ret = insert_trim(of->start_time, of->recording_time,

                      &last_filter, &pad_idx, name);

    if (ret < 0)

        return ret;





    if ((ret = avfilter_link(last_filter, pad_idx, ofilter->filter, 0)) < 0)

        return ret;



    return 0;

}
