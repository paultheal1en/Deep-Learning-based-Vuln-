static int configure_filters(AVInputStream *ist, AVOutputStream *ost)

{

    AVFilterContext *curr_filter;

    /** filter graph containing all filters including input & output */

    AVCodecContext *codec = ost->st->codec;

    AVCodecContext *icodec = ist->st->codec;

    char args[255];



    filt_graph_all = av_mallocz(sizeof(AVFilterGraph));



    if(!(ist->input_video_filter = avfilter_open(avfilter_get_by_name("buffer"), "src")))

        return -1;

    if(!(ist->out_video_filter = avfilter_open(&output_filter, "out")))

        return -1;



    snprintf(args, 255, "%d:%d:%d", ist->st->codec->width,

             ist->st->codec->height, ist->st->codec->pix_fmt);

    if(avfilter_init_filter(ist->input_video_filter, args, NULL))

        return -1;

    if(avfilter_init_filter(ist->out_video_filter, NULL, &codec->pix_fmt))

        return -1;



    /* add input and output filters to the overall graph */

    avfilter_graph_add_filter(filt_graph_all, ist->input_video_filter);

    avfilter_graph_add_filter(filt_graph_all, ist->out_video_filter);



    curr_filter = ist->input_video_filter;



    if(ost->video_crop) {

        char crop_args[255];

        AVFilterContext *filt_crop;

        snprintf(crop_args, 255, "%d:%d:%d:%d", ost->leftBand, ost->topBand,

                 codec->width -  (frame_padleft + frame_padright),

                 codec->height - (frame_padtop + frame_padbottom));

        filt_crop = avfilter_open(avfilter_get_by_name("crop"), NULL);

        if (!filt_crop)

            return -1;

        if (avfilter_init_filter(filt_crop, crop_args, NULL))

            return -1;

        if (avfilter_link(curr_filter, 0, filt_crop, 0))

            return -1;

        curr_filter = filt_crop;

        avfilter_graph_add_filter(filt_graph_all, curr_filter);

    }



    if((codec->width !=

        icodec->width - (frame_leftBand + frame_rightBand) +

        (frame_padleft + frame_padright)) ||

       (codec->height != icodec->height - (frame_topBand  + frame_bottomBand) +

        (frame_padtop + frame_padbottom))) {

        char scale_args[255];

        AVFilterContext *filt_scale;

        snprintf(scale_args, 255, "%d:%d:flags=0x%X",

                 codec->width  - (frame_padleft + frame_padright),

                 codec->height - (frame_padtop  + frame_padbottom),

                 (int)av_get_int(sws_opts, "sws_flags", NULL));

        filt_scale = avfilter_open(avfilter_get_by_name("scale"), NULL);

        if (!filt_scale)

            return -1;

        if (avfilter_init_filter(filt_scale, scale_args, NULL))

            return -1;

        if (avfilter_link(curr_filter, 0, filt_scale, 0))

            return -1;

        curr_filter = filt_scale;

        avfilter_graph_add_filter(filt_graph_all, curr_filter);

    }



    if(vfilters) {

        AVFilterInOut *outputs = av_malloc(sizeof(AVFilterInOut));

        AVFilterInOut *inputs  = av_malloc(sizeof(AVFilterInOut));



        outputs->name    = av_strdup("in");

        outputs->filter  = curr_filter;

        outputs->pad_idx = 0;

        outputs->next    = NULL;



        inputs->name    = av_strdup("out");

        inputs->filter  = ist->out_video_filter;

        inputs->pad_idx = 0;

        inputs->next    = NULL;



        if (avfilter_graph_parse(filt_graph_all, vfilters, inputs, outputs, NULL) < 0)

            return -1;

        av_freep(&vfilters);

    } else {

        if(avfilter_link(curr_filter, 0, ist->out_video_filter, 0) < 0)

            return -1;

    }



    {

        char scale_sws_opts[128];

        snprintf(scale_sws_opts, sizeof(scale_sws_opts), "flags=0x%X", (int)av_get_int(sws_opts, "sws_flags", NULL));

        filt_graph_all->scale_sws_opts = av_strdup(scale_sws_opts);

    }



    /* configure all the filter links */

    if(avfilter_graph_check_validity(filt_graph_all, NULL))

        return -1;

    if(avfilter_graph_config_formats(filt_graph_all, NULL))

        return -1;

    if(avfilter_graph_config_links(filt_graph_all, NULL))

        return -1;



    codec->width = ist->out_video_filter->inputs[0]->w;

    codec->height = ist->out_video_filter->inputs[0]->h;



    return 0;

}
