static OutputStream *new_output_stream(OptionsContext *o, AVFormatContext *oc, enum AVMediaType type)

{

    OutputStream *ost;

    AVStream *st = avformat_new_stream(oc, NULL);

    int idx      = oc->nb_streams - 1, ret = 0;

    char *bsf = NULL, *next, *codec_tag = NULL;

    AVBitStreamFilterContext *bsfc, *bsfc_prev = NULL;

    double qscale = -1;

    char *buf = NULL, *arg = NULL, *preset = NULL;

    AVIOContext *s = NULL;



    if (!st) {

        av_log(NULL, AV_LOG_FATAL, "Could not alloc stream.\n");

        exit(1);

    }



    if (oc->nb_streams - 1 < o->nb_streamid_map)

        st->id = o->streamid_map[oc->nb_streams - 1];



    output_streams = grow_array(output_streams, sizeof(*output_streams), &nb_output_streams,

                                nb_output_streams + 1);

    if (!(ost = av_mallocz(sizeof(*ost))))

        exit(1);

    output_streams[nb_output_streams - 1] = ost;



    ost->file_index = nb_output_files;

    ost->index      = idx;

    ost->st         = st;

    st->codec->codec_type = type;

    choose_encoder(o, oc, ost);

    if (ost->enc) {

        ost->opts  = filter_codec_opts(codec_opts, ost->enc->id, oc, st, ost->enc);

    }



    avcodec_get_context_defaults3(st->codec, ost->enc);

    st->codec->codec_type = type; // XXX hack, avcodec_get_context_defaults2() sets type to unknown for stream copy



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

                exit(1);

            }

            *arg++ = 0;

            av_dict_set(&ost->opts, buf, arg, AV_DICT_DONT_OVERWRITE);

            av_free(buf);

        } while (!s->eof_reached);

        avio_close(s);

    }

    if (ret) {

        av_log(NULL, AV_LOG_FATAL,

               "Preset %s specified for stream %d:%d, but could not be opened.\n",

               preset, ost->file_index, ost->index);

        exit(1);

    }



    ost->max_frames = INT64_MAX;

    MATCH_PER_STREAM_OPT(max_frames, i64, ost->max_frames, oc, st);



    MATCH_PER_STREAM_OPT(bitstream_filters, str, bsf, oc, st);

    while (bsf) {

        if (next = strchr(bsf, ','))

            *next++ = 0;

        if (!(bsfc = av_bitstream_filter_init(bsf))) {

            av_log(NULL, AV_LOG_FATAL, "Unknown bitstream filter %s\n", bsf);

            exit(1);

        }

        if (bsfc_prev)

            bsfc_prev->next = bsfc;

        else

            ost->bitstream_filters = bsfc;



        bsfc_prev = bsfc;

        bsf       = next;

    }



    MATCH_PER_STREAM_OPT(codec_tags, str, codec_tag, oc, st);

    if (codec_tag) {

        uint32_t tag = strtol(codec_tag, &next, 0);

        if (*next)

            tag = AV_RL32(codec_tag);

        st->codec->codec_tag = tag;

    }



    MATCH_PER_STREAM_OPT(qscale, dbl, qscale, oc, st);

    if (qscale >= 0) {

        st->codec->flags |= CODEC_FLAG_QSCALE;

        st->codec->global_quality = FF_QP2LAMBDA * qscale;

    }



    if (oc->oformat->flags & AVFMT_GLOBALHEADER)

        st->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;



    av_opt_get_int(sws_opts, "sws_flags", 0, &ost->sws_flags);



    ost->pix_fmts[0] = ost->pix_fmts[1] = AV_PIX_FMT_NONE;



    return ost;

}
