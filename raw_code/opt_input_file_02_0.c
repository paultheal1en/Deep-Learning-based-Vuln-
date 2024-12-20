static int opt_input_file(OptionsContext *o, const char *opt, const char *filename)

{

    AVFormatContext *ic;

    AVInputFormat *file_iformat = NULL;

    int err, i, ret;

    int64_t timestamp;

    uint8_t buf[128];

    AVDictionary **opts;

    int orig_nb_streams;                     // number of streams before avformat_find_stream_info



    if (o->format) {

        if (!(file_iformat = av_find_input_format(o->format))) {

            av_log(NULL, AV_LOG_FATAL, "Unknown input format: '%s'\n", o->format);

            exit_program(1);

        }

    }



    if (!strcmp(filename, "-"))

        filename = "pipe:";



    using_stdin |= !strncmp(filename, "pipe:", 5) ||

                    !strcmp(filename, "/dev/stdin");



    /* get default parameters from command line */

    ic = avformat_alloc_context();

    if (!ic) {

        print_error(filename, AVERROR(ENOMEM));

        exit_program(1);

    }

    if (o->nb_audio_sample_rate) {

        snprintf(buf, sizeof(buf), "%d", o->audio_sample_rate[o->nb_audio_sample_rate - 1].u.i);

        av_dict_set(&format_opts, "sample_rate", buf, 0);

    }

    if (o->nb_audio_channels) {

        snprintf(buf, sizeof(buf), "%d", o->audio_channels[o->nb_audio_channels - 1].u.i);

        av_dict_set(&format_opts, "channels", buf, 0);

    }

    if (o->nb_frame_rates) {

        av_dict_set(&format_opts, "framerate", o->frame_rates[o->nb_frame_rates - 1].u.str, 0);

    }

    if (o->nb_frame_sizes) {

        av_dict_set(&format_opts, "video_size", o->frame_sizes[o->nb_frame_sizes - 1].u.str, 0);

    }

    if (o->nb_frame_pix_fmts)

        av_dict_set(&format_opts, "pixel_format", o->frame_pix_fmts[o->nb_frame_pix_fmts - 1].u.str, 0);



    ic->video_codec_id   = video_codec_name ?

        find_codec_or_die(video_codec_name   , AVMEDIA_TYPE_VIDEO   , 0)->id : CODEC_ID_NONE;

    ic->audio_codec_id   = audio_codec_name ?

        find_codec_or_die(audio_codec_name   , AVMEDIA_TYPE_AUDIO   , 0)->id : CODEC_ID_NONE;

    ic->subtitle_codec_id= subtitle_codec_name ?

        find_codec_or_die(subtitle_codec_name, AVMEDIA_TYPE_SUBTITLE, 0)->id : CODEC_ID_NONE;

    ic->flags |= AVFMT_FLAG_NONBLOCK;

    ic->interrupt_callback = int_cb;



    if (loop_input) {

        av_log(NULL, AV_LOG_WARNING, "-loop_input is deprecated, use -loop 1\n");

        ic->loop_input = loop_input;

    }



    /* open the input file with generic avformat function */

    err = avformat_open_input(&ic, filename, file_iformat, &format_opts);

    if (err < 0) {

        print_error(filename, err);

        exit_program(1);

    }

    assert_avoptions(format_opts);



    /* apply forced codec ids */

    for (i = 0; i < ic->nb_streams; i++)

        choose_decoder(o, ic, ic->streams[i]);



    /* Set AVCodecContext options for avformat_find_stream_info */

    opts = setup_find_stream_info_opts(ic, codec_opts);

    orig_nb_streams = ic->nb_streams;



    /* If not enough info to get the stream parameters, we decode the

       first frames to get it. (used in mpeg case for example) */

    ret = avformat_find_stream_info(ic, opts);

    if (ret < 0) {

        av_log(NULL, AV_LOG_FATAL, "%s: could not find codec parameters\n", filename);

        av_close_input_file(ic);

        exit_program(1);

    }



    timestamp = o->start_time;

    /* add the stream start time */

    if (ic->start_time != AV_NOPTS_VALUE)

        timestamp += ic->start_time;



    /* if seeking requested, we execute it */

    if (o->start_time != 0) {

        ret = av_seek_frame(ic, -1, timestamp, AVSEEK_FLAG_BACKWARD);

        if (ret < 0) {

            av_log(NULL, AV_LOG_WARNING, "%s: could not seek to position %0.3f\n",

                   filename, (double)timestamp / AV_TIME_BASE);

        }

    }



    /* update the current parameters so that they match the one of the input stream */

    add_input_streams(o, ic);



    /* dump the file content */

    av_dump_format(ic, nb_input_files, filename, 0);



    input_files = grow_array(input_files, sizeof(*input_files), &nb_input_files, nb_input_files + 1);

    input_files[nb_input_files - 1].ctx        = ic;

    input_files[nb_input_files - 1].ist_index  = nb_input_streams - ic->nb_streams;

    input_files[nb_input_files - 1].ts_offset  = o->input_ts_offset - (copy_ts ? 0 : timestamp);

    input_files[nb_input_files - 1].nb_streams = ic->nb_streams;

    input_files[nb_input_files - 1].rate_emu   = o->rate_emu;



    for (i = 0; i < o->nb_dump_attachment; i++) {

        int j;



        for (j = 0; j < ic->nb_streams; j++) {

            AVStream *st = ic->streams[j];



            if (check_stream_specifier(ic, st, o->dump_attachment[i].specifier) == 1)

                dump_attachment(st, o->dump_attachment[i].u.str);

        }

    }



    for (i = 0; i < orig_nb_streams; i++)

        av_dict_free(&opts[i]);

    av_freep(&opts);



    reset_options(o, 1);

    return 0;

}
