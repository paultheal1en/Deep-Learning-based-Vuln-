static OutputStream *new_video_stream(OptionsContext *o, AVFormatContext *oc, int source_index)

{

    AVStream *st;

    OutputStream *ost;

    AVCodecContext *video_enc;

    char *frame_rate = NULL, *frame_aspect_ratio = NULL;



    ost = new_output_stream(o, oc, AVMEDIA_TYPE_VIDEO, source_index);

    st  = ost->st;

    video_enc = ost->enc_ctx;



    MATCH_PER_STREAM_OPT(frame_rates, str, frame_rate, oc, st);

    if (frame_rate && av_parse_video_rate(&ost->frame_rate, frame_rate) < 0) {

        av_log(NULL, AV_LOG_FATAL, "Invalid framerate value: %s\n", frame_rate);

        exit_program(1);

    }

    if (frame_rate && video_sync_method == VSYNC_PASSTHROUGH)

        av_log(NULL, AV_LOG_ERROR, "Using -vsync 0 and -r can produce invalid output files\n");



    MATCH_PER_STREAM_OPT(frame_aspect_ratios, str, frame_aspect_ratio, oc, st);

    if (frame_aspect_ratio) {

        AVRational q;

        if (av_parse_ratio(&q, frame_aspect_ratio, 255, 0, NULL) < 0 ||

            q.num <= 0 || q.den <= 0) {

            av_log(NULL, AV_LOG_FATAL, "Invalid aspect ratio: %s\n", frame_aspect_ratio);

            exit_program(1);

        }

        ost->frame_aspect_ratio = q;

    }



    MATCH_PER_STREAM_OPT(filter_scripts, str, ost->filters_script, oc, st);

    MATCH_PER_STREAM_OPT(filters,        str, ost->filters,        oc, st);



    if (!ost->stream_copy) {

        const char *p = NULL;

        char *frame_size = NULL;

        char *frame_pix_fmt = NULL;

        char *intra_matrix = NULL, *inter_matrix = NULL;

        char *chroma_intra_matrix = NULL;

        int do_pass = 0;

        int i;



        MATCH_PER_STREAM_OPT(frame_sizes, str, frame_size, oc, st);

        if (frame_size && av_parse_video_size(&video_enc->width, &video_enc->height, frame_size) < 0) {

            av_log(NULL, AV_LOG_FATAL, "Invalid frame size: %s.\n", frame_size);

            exit_program(1);

        }



        video_enc->bits_per_raw_sample = frame_bits_per_raw_sample;

        MATCH_PER_STREAM_OPT(frame_pix_fmts, str, frame_pix_fmt, oc, st);

        if (frame_pix_fmt && *frame_pix_fmt == '+') {

            ost->keep_pix_fmt = 1;

            if (!*++frame_pix_fmt)

                frame_pix_fmt = NULL;

        }

        if (frame_pix_fmt && (video_enc->pix_fmt = av_get_pix_fmt(frame_pix_fmt)) == AV_PIX_FMT_NONE) {

            av_log(NULL, AV_LOG_FATAL, "Unknown pixel format requested: %s.\n", frame_pix_fmt);

            exit_program(1);

        }

        st->sample_aspect_ratio = video_enc->sample_aspect_ratio;



        if (intra_only)

            video_enc->gop_size = 0;

        MATCH_PER_STREAM_OPT(intra_matrices, str, intra_matrix, oc, st);

        if (intra_matrix) {

            if (!(video_enc->intra_matrix = av_mallocz(sizeof(*video_enc->intra_matrix) * 64))) {

                av_log(NULL, AV_LOG_FATAL, "Could not allocate memory for intra matrix.\n");

                exit_program(1);

            }

            parse_matrix_coeffs(video_enc->intra_matrix, intra_matrix);

        }

        MATCH_PER_STREAM_OPT(chroma_intra_matrices, str, chroma_intra_matrix, oc, st);

        if (chroma_intra_matrix) {

            uint16_t *p = av_mallocz(sizeof(*video_enc->chroma_intra_matrix) * 64);

            if (!p) {

                av_log(NULL, AV_LOG_FATAL, "Could not allocate memory for intra matrix.\n");

                exit_program(1);

            }

            av_codec_set_chroma_intra_matrix(video_enc, p);

            parse_matrix_coeffs(p, chroma_intra_matrix);

        }

        MATCH_PER_STREAM_OPT(inter_matrices, str, inter_matrix, oc, st);

        if (inter_matrix) {

            if (!(video_enc->inter_matrix = av_mallocz(sizeof(*video_enc->inter_matrix) * 64))) {

                av_log(NULL, AV_LOG_FATAL, "Could not allocate memory for inter matrix.\n");

                exit_program(1);

            }

            parse_matrix_coeffs(video_enc->inter_matrix, inter_matrix);

        }



        MATCH_PER_STREAM_OPT(rc_overrides, str, p, oc, st);

        for (i = 0; p; i++) {

            int start, end, q;

            int e = sscanf(p, "%d,%d,%d", &start, &end, &q);

            if (e != 3) {

                av_log(NULL, AV_LOG_FATAL, "error parsing rc_override\n");

                exit_program(1);

            }

            /* FIXME realloc failure */

            video_enc->rc_override =

                av_realloc_array(video_enc->rc_override,

                                 i + 1, sizeof(RcOverride));

            video_enc->rc_override[i].start_frame = start;

            video_enc->rc_override[i].end_frame   = end;

            if (q > 0) {

                video_enc->rc_override[i].qscale         = q;

                video_enc->rc_override[i].quality_factor = 1.0;

            }

            else {

                video_enc->rc_override[i].qscale         = 0;

                video_enc->rc_override[i].quality_factor = -q/100.0;

            }

            p = strchr(p, '/');

            if (p) p++;

        }

        video_enc->rc_override_count = i;



        if (do_psnr)

            video_enc->flags|= CODEC_FLAG_PSNR;



        /* two pass mode */

        MATCH_PER_STREAM_OPT(pass, i, do_pass, oc, st);

        if (do_pass) {

            if (do_pass & 1) {

                video_enc->flags |= CODEC_FLAG_PASS1;

                av_dict_set(&ost->encoder_opts, "flags", "+pass1", AV_DICT_APPEND);

            }

            if (do_pass & 2) {

                video_enc->flags |= CODEC_FLAG_PASS2;

                av_dict_set(&ost->encoder_opts, "flags", "+pass2", AV_DICT_APPEND);

            }

        }



        MATCH_PER_STREAM_OPT(passlogfiles, str, ost->logfile_prefix, oc, st);

        if (ost->logfile_prefix &&

            !(ost->logfile_prefix = av_strdup(ost->logfile_prefix)))

            exit_program(1);



        MATCH_PER_STREAM_OPT(forced_key_frames, str, ost->forced_keyframes, oc, st);

        if (ost->forced_keyframes)

            ost->forced_keyframes = av_strdup(ost->forced_keyframes);



        MATCH_PER_STREAM_OPT(force_fps, i, ost->force_fps, oc, st);



        ost->top_field_first = -1;

        MATCH_PER_STREAM_OPT(top_field_first, i, ost->top_field_first, oc, st);





        ost->avfilter = get_ost_filters(o, oc, ost);

        if (!ost->avfilter)

            exit_program(1);

    } else {

        MATCH_PER_STREAM_OPT(copy_initial_nonkeyframes, i, ost->copy_initial_nonkeyframes, oc ,st);

    }



    if (ost->stream_copy)

        check_streamcopy_filters(o, oc, ost, AVMEDIA_TYPE_VIDEO);



    return ost;

}
