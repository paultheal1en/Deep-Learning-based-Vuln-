static OutputStream *new_output_stream(OptionsContext *o, AVFormatContext *oc, enum AVMediaType type, int source_index)

{

    OutputStream *ost;

    AVStream *st = avformat_new_stream(oc, NULL);

    int idx      = oc->nb_streams - 1, ret = 0;

    const char *bsfs = NULL, *time_base = NULL;

    char *next, *codec_tag = NULL;

    double qscale = -1;

    int i;



    if (!st) {

        av_log(NULL, AV_LOG_FATAL, "Could not alloc stream.\n");

        exit_program(1);

    }



    if (oc->nb_streams - 1 < o->nb_streamid_map)

        st->id = o->streamid_map[oc->nb_streams - 1];



    GROW_ARRAY(output_streams, nb_output_streams);

    if (!(ost = av_mallocz(sizeof(*ost))))

        exit_program(1);

    output_streams[nb_output_streams - 1] = ost;



    ost->file_index = nb_output_files - 1;

    ost->index      = idx;

    ost->st         = st;

    st->codecpar->codec_type = type;



    ret = choose_encoder(o, oc, ost);

    if (ret < 0) {

        av_log(NULL, AV_LOG_FATAL, "Error selecting an encoder for stream "

               "%d:%d\n", ost->file_index, ost->index);

        exit_program(1);

    }



    ost->enc_ctx = avcodec_alloc_context3(ost->enc);

    if (!ost->enc_ctx) {

        av_log(NULL, AV_LOG_ERROR, "Error allocating the encoding context.\n");

        exit_program(1);

    }

    ost->enc_ctx->codec_type = type;



    ost->ref_par = avcodec_parameters_alloc();

    if (!ost->ref_par) {

        av_log(NULL, AV_LOG_ERROR, "Error allocating the encoding parameters.\n");

        exit_program(1);

    }



    if (ost->enc) {

        AVIOContext *s = NULL;

        char *buf = NULL, *arg = NULL, *preset = NULL;



        ost->encoder_opts  = filter_codec_opts(o->g->codec_opts, ost->enc->id, oc, st, ost->enc);



        MATCH_PER_STREAM_OPT(presets, str, preset, oc, st);

        if (preset && (!(ret = get_preset_file_2(preset, ost->enc->name, &s)))) {

            do  {

                buf = get_line(s);

                if (!buf[0] || buf[0] == '#') {

                    av_free(buf);

                    continue;

                }

                if (!(arg = strchr(buf, '='))) {

                    av_log(NULL, AV_LOG_FATAL, "Invalid line found in the preset file.\n");

                    exit_program(1);

                }

                *arg++ = 0;

                av_dict_set(&ost->encoder_opts, buf, arg, AV_DICT_DONT_OVERWRITE);

                av_free(buf);

            } while (!s->eof_reached);

            avio_closep(&s);

        }

        if (ret) {

            av_log(NULL, AV_LOG_FATAL,

                   "Preset %s specified for stream %d:%d, but could not be opened.\n",

                   preset, ost->file_index, ost->index);

            exit_program(1);

        }

    } else {

        ost->encoder_opts = filter_codec_opts(o->g->codec_opts, AV_CODEC_ID_NONE, oc, st, NULL);

    }



    MATCH_PER_STREAM_OPT(time_bases, str, time_base, oc, st);

    if (time_base) {

        AVRational q;

        if (av_parse_ratio(&q, time_base, INT_MAX, 0, NULL) < 0 ||

            q.num <= 0 || q.den <= 0) {

            av_log(NULL, AV_LOG_FATAL, "Invalid time base: %s\n", time_base);

            exit_program(1);

        }

        st->time_base = q;

    }



    MATCH_PER_STREAM_OPT(enc_time_bases, str, time_base, oc, st);

    if (time_base) {

        AVRational q;

        if (av_parse_ratio(&q, time_base, INT_MAX, 0, NULL) < 0 ||

            q.den <= 0) {

            av_log(NULL, AV_LOG_FATAL, "Invalid time base: %s\n", time_base);

            exit_program(1);

        }

        ost->enc_timebase = q;

    }



    ost->max_frames = INT64_MAX;

    MATCH_PER_STREAM_OPT(max_frames, i64, ost->max_frames, oc, st);

    for (i = 0; i<o->nb_max_frames; i++) {

        char *p = o->max_frames[i].specifier;

        if (!*p && type != AVMEDIA_TYPE_VIDEO) {

            av_log(NULL, AV_LOG_WARNING, "Applying unspecific -frames to non video streams, maybe you meant -vframes ?\n");

            break;

        }

    }



    ost->copy_prior_start = -1;

    MATCH_PER_STREAM_OPT(copy_prior_start, i, ost->copy_prior_start, oc ,st);



    MATCH_PER_STREAM_OPT(bitstream_filters, str, bsfs, oc, st);

    while (bsfs && *bsfs) {

        const AVBitStreamFilter *filter;

        char *bsf, *bsf_options_str, *bsf_name;



        bsf = av_get_token(&bsfs, ",");

        if (!bsf)

            exit_program(1);

        bsf_name = av_strtok(bsf, "=", &bsf_options_str);

        if (!bsf_name)

            exit_program(1);



        filter = av_bsf_get_by_name(bsf_name);

        if (!filter) {

            av_log(NULL, AV_LOG_FATAL, "Unknown bitstream filter %s\n", bsf_name);

            exit_program(1);

        }



        ost->bsf_ctx = av_realloc_array(ost->bsf_ctx,

                                        ost->nb_bitstream_filters + 1,

                                        sizeof(*ost->bsf_ctx));

        if (!ost->bsf_ctx)

            exit_program(1);



        ret = av_bsf_alloc(filter, &ost->bsf_ctx[ost->nb_bitstream_filters]);

        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error allocating a bitstream filter context\n");

            exit_program(1);

        }



        ost->nb_bitstream_filters++;



        if (bsf_options_str && filter->priv_class) {

            const AVOption *opt = av_opt_next(ost->bsf_ctx[ost->nb_bitstream_filters-1]->priv_data, NULL);

            const char * shorthand[2] = {NULL};



            if (opt)

                shorthand[0] = opt->name;



            ret = av_opt_set_from_string(ost->bsf_ctx[ost->nb_bitstream_filters-1]->priv_data, bsf_options_str, shorthand, "=", ":");

            if (ret < 0) {

                av_log(NULL, AV_LOG_ERROR, "Error parsing options for bitstream filter %s\n", bsf_name);

                exit_program(1);

            }

        }

        av_freep(&bsf);



        if (*bsfs)

            bsfs++;

    }

    if (ost->nb_bitstream_filters) {

        ost->bsf_extradata_updated = av_mallocz_array(ost->nb_bitstream_filters, sizeof(*ost->bsf_extradata_updated));

        if (!ost->bsf_extradata_updated) {

            av_log(NULL, AV_LOG_FATAL, "Bitstream filter memory allocation failed\n");

            exit_program(1);

        }

    }



    MATCH_PER_STREAM_OPT(codec_tags, str, codec_tag, oc, st);

    if (codec_tag) {

        uint32_t tag = strtol(codec_tag, &next, 0);

        if (*next)

            tag = AV_RL32(codec_tag);

        ost->st->codecpar->codec_tag =

        ost->enc_ctx->codec_tag = tag;

    }



    MATCH_PER_STREAM_OPT(qscale, dbl, qscale, oc, st);

    if (qscale >= 0) {

        ost->enc_ctx->flags |= AV_CODEC_FLAG_QSCALE;

        ost->enc_ctx->global_quality = FF_QP2LAMBDA * qscale;

    }



    MATCH_PER_STREAM_OPT(disposition, str, ost->disposition, oc, st);

    ost->disposition = av_strdup(ost->disposition);



    ost->max_muxing_queue_size = 128;

    MATCH_PER_STREAM_OPT(max_muxing_queue_size, i, ost->max_muxing_queue_size, oc, st);

    ost->max_muxing_queue_size *= sizeof(AVPacket);



    if (oc->oformat->flags & AVFMT_GLOBALHEADER)

        ost->enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;



    av_dict_copy(&ost->sws_dict, o->g->sws_dict, 0);



    av_dict_copy(&ost->swr_opts, o->g->swr_opts, 0);

    if (ost->enc && av_get_exact_bits_per_sample(ost->enc->id) == 24)

        av_dict_set(&ost->swr_opts, "output_sample_bits", "24", 0);



    av_dict_copy(&ost->resample_opts, o->g->resample_opts, 0);



    ost->source_index = source_index;

    if (source_index >= 0) {

        ost->sync_ist = input_streams[source_index];

        input_streams[source_index]->discard = 0;

        input_streams[source_index]->st->discard = input_streams[source_index]->user_set_discard;

    }

    ost->last_mux_dts = AV_NOPTS_VALUE;



    ost->muxing_queue = av_fifo_alloc(8 * sizeof(AVPacket));

    if (!ost->muxing_queue)

        exit_program(1);



    return ost;

}
