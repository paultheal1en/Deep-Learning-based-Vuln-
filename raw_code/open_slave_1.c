static int open_slave(AVFormatContext *avf, char *slave, TeeSlave *tee_slave)

{

    int i, ret;

    AVDictionary *options = NULL;

    AVDictionaryEntry *entry;

    char *filename;

    char *format = NULL, *select = NULL;

    AVFormatContext *avf2 = NULL;

    AVStream *st, *st2;

    int stream_count;

    int fullret;

    char *subselect = NULL, *next_subselect = NULL, *first_subselect = NULL, *tmp_select = NULL;



    if ((ret = parse_slave_options(avf, slave, &options, &filename)) < 0)

        return ret;



#define STEAL_OPTION(option, field) do {                                \

        if ((entry = av_dict_get(options, option, NULL, 0))) {          \

            field = entry->value;                                       \

            entry->value = NULL; /* prevent it from being freed */      \

            av_dict_set(&options, option, NULL, 0);                     \

        }                                                               \

    } while (0)



    STEAL_OPTION("f", format);

    STEAL_OPTION("select", select);



    ret = avformat_alloc_output_context2(&avf2, NULL, format, filename);

    if (ret < 0)

        goto end;

    av_dict_copy(&avf2->metadata, avf->metadata, 0);

    avf2->opaque   = avf->opaque;

    avf2->io_open  = avf->io_open;

    avf2->io_close = avf->io_close;



    tee_slave->stream_map = av_calloc(avf->nb_streams, sizeof(*tee_slave->stream_map));

    if (!tee_slave->stream_map) {

        ret = AVERROR(ENOMEM);

        goto end;

    }



    stream_count = 0;

    for (i = 0; i < avf->nb_streams; i++) {

        st = avf->streams[i];

        if (select) {

            tmp_select = av_strdup(select);  // av_strtok is destructive so we regenerate it in each loop

            if (!tmp_select) {

                ret = AVERROR(ENOMEM);

                goto end;

            }

            fullret = 0;

            first_subselect = tmp_select;

            next_subselect = NULL;

            while (subselect = av_strtok(first_subselect, slave_select_sep, &next_subselect)) {

                first_subselect = NULL;



                ret = avformat_match_stream_specifier(avf, avf->streams[i], subselect);

                if (ret < 0) {

                    av_log(avf, AV_LOG_ERROR,

                           "Invalid stream specifier '%s' for output '%s'\n",

                           subselect, slave);

                    goto end;

                }

                if (ret != 0) {

                    fullret = 1; // match

                    break;

                }

            }

            av_freep(&tmp_select);



            if (fullret == 0) { /* no match */

                tee_slave->stream_map[i] = -1;

                continue;

            }

        }

        tee_slave->stream_map[i] = stream_count++;



        if (!(st2 = avformat_new_stream(avf2, NULL))) {

            ret = AVERROR(ENOMEM);

            goto end;

        }

        st2->id = st->id;

        st2->r_frame_rate        = st->r_frame_rate;

        st2->time_base           = st->time_base;

        st2->start_time          = st->start_time;

        st2->duration            = st->duration;

        st2->nb_frames           = st->nb_frames;

        st2->disposition         = st->disposition;

        st2->sample_aspect_ratio = st->sample_aspect_ratio;

        st2->avg_frame_rate      = st->avg_frame_rate;

        av_dict_copy(&st2->metadata, st->metadata, 0);

        if ((ret = avcodec_parameters_copy(st2->codecpar, st->codecpar)) < 0)

            goto end;

    }



    if (!(avf2->oformat->flags & AVFMT_NOFILE)) {

        if ((ret = avf2->io_open(avf2, &avf2->pb, filename, AVIO_FLAG_WRITE, NULL)) < 0) {

            av_log(avf, AV_LOG_ERROR, "Slave '%s': error opening: %s\n",

                   slave, av_err2str(ret));

            goto end;

        }

    }



    if ((ret = avformat_write_header(avf2, &options)) < 0) {

        av_log(avf, AV_LOG_ERROR, "Slave '%s': error writing header: %s\n",

               slave, av_err2str(ret));

        goto end;

    }



    tee_slave->avf = avf2;

    tee_slave->bsfs = av_calloc(avf2->nb_streams, sizeof(TeeSlave));

    if (!tee_slave->bsfs) {

        ret = AVERROR(ENOMEM);

        goto end;

    }



    entry = NULL;

    while (entry = av_dict_get(options, "bsfs", NULL, AV_DICT_IGNORE_SUFFIX)) {

        const char *spec = entry->key + strlen("bsfs");

        if (*spec) {

            if (strspn(spec, slave_bsfs_spec_sep) != 1) {

                av_log(avf, AV_LOG_ERROR,

                       "Specifier separator in '%s' is '%c', but only characters '%s' "

                       "are allowed\n", entry->key, *spec, slave_bsfs_spec_sep);

                return AVERROR(EINVAL);

            }

            spec++; /* consume separator */

        }



        for (i = 0; i < avf2->nb_streams; i++) {

            ret = avformat_match_stream_specifier(avf2, avf2->streams[i], spec);

            if (ret < 0) {

                av_log(avf, AV_LOG_ERROR,

                       "Invalid stream specifier '%s' in bsfs option '%s' for slave "

                       "output '%s'\n", spec, entry->key, filename);

                goto end;

            }



            if (ret > 0) {

                av_log(avf, AV_LOG_DEBUG, "spec:%s bsfs:%s matches stream %d of slave "

                       "output '%s'\n", spec, entry->value, i, filename);

                if (tee_slave->bsfs[i]) {

                    av_log(avf, AV_LOG_WARNING,

                           "Duplicate bsfs specification associated to stream %d of slave "

                           "output '%s', filters will be ignored\n", i, filename);

                    continue;

                }

                ret = parse_bsfs(avf, entry->value, &tee_slave->bsfs[i]);

                if (ret < 0) {

                    av_log(avf, AV_LOG_ERROR,

                           "Error parsing bitstream filter sequence '%s' associated to "

                           "stream %d of slave output '%s'\n", entry->value, i, filename);

                    goto end;

                }

            }

        }



        av_dict_set(&options, entry->key, NULL, 0);

    }



    if (options) {

        entry = NULL;

        while ((entry = av_dict_get(options, "", entry, AV_DICT_IGNORE_SUFFIX)))

            av_log(avf2, AV_LOG_ERROR, "Unknown option '%s'\n", entry->key);

        ret = AVERROR_OPTION_NOT_FOUND;

        goto end;

    }



end:

    av_free(format);

    av_free(select);

    av_dict_free(&options);

    av_freep(&tmp_select);

    return ret;

}
