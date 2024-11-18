static OutputStream *new_output_stream(OptionsContext *o, AVFormatContext *oc, enum AVMediaType type, int source_index)

{

    OutputStream *ost;

    AVStream *st = avformat_new_stream(oc, NULL);

    int idx      = oc->nb_streams - 1, ret = 0;

    char *bsf = NULL, *next, *codec_tag = NULL;

    AVBitStreamFilterContext *bsfc, *bsfc_prev = NULL;

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



    MATCH_PER_STREAM_OPT(bitstream_filters, str, bsf, oc, st);

    while (bsf) {

        char *arg = NULL;

        if (next = strchr(bsf, ','))

            *next++ = 0;

        if (arg = strchr(bsf, '='))

            *arg++ = 0;

        if (!(bsfc = av_bitstream_filter_init(bsf))) {

            av_log(NULL, AV_LOG_FATAL, "Unknown bitstream filter %s\n", bsf);

            exit_program(1);

        }

        if (bsfc_prev)

            bsfc_prev->next = bsfc;

        else

            ost->bitstream_filters = bsfc;

        if (arg)

            if (!(bsfc->args = av_strdup(arg))) {

                av_log(NULL, AV_LOG_FATAL, "Bitstream filter memory allocation failed\n");

                exit_program(1);

            }



        bsfc_prev = bsfc;

        bsf       = next;

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



    return ost;

}
