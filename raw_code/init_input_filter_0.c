static void init_input_filter(FilterGraph *fg, AVFilterInOut *in)

{

    InputStream *ist = NULL;

    enum AVMediaType type = avfilter_pad_get_type(in->filter_ctx->input_pads, in->pad_idx);

    int i;



    // TODO: support other filter types

    if (type != AVMEDIA_TYPE_VIDEO && type != AVMEDIA_TYPE_AUDIO) {

        av_log(NULL, AV_LOG_FATAL, "Only video and audio filters supported "

               "currently.\n");

        exit(1);

    }



    if (in->name) {

        AVFormatContext *s;

        AVStream       *st = NULL;

        char *p;

        int file_idx = strtol(in->name, &p, 0);



        if (file_idx < 0 || file_idx >= nb_input_files) {

            av_log(NULL, AV_LOG_FATAL, "Invalid file index %d in filtegraph description %s.\n",

                   file_idx, fg->graph_desc);

            exit(1);

        }

        s = input_files[file_idx]->ctx;



        for (i = 0; i < s->nb_streams; i++) {

            if (s->streams[i]->codecpar->codec_type != type)

                continue;

            if (check_stream_specifier(s, s->streams[i], *p == ':' ? p + 1 : p) == 1) {

                st = s->streams[i];

                break;

            }

        }

        if (!st) {

            av_log(NULL, AV_LOG_FATAL, "Stream specifier '%s' in filtergraph description %s "

                   "matches no streams.\n", p, fg->graph_desc);

            exit(1);

        }

        ist = input_streams[input_files[file_idx]->ist_index + st->index];

    } else {

        /* find the first unused stream of corresponding type */

        for (i = 0; i < nb_input_streams; i++) {

            ist = input_streams[i];

            if (ist->dec_ctx->codec_type == type && ist->discard)

                break;

        }

        if (i == nb_input_streams) {

            av_log(NULL, AV_LOG_FATAL, "Cannot find a matching stream for "

                   "unlabeled input pad %d on filter %s\n", in->pad_idx,

                   in->filter_ctx->name);

            exit(1);

        }

    }

    av_assert0(ist);



    ist->discard         = 0;

    ist->decoding_needed = 1;

    ist->st->discard = AVDISCARD_NONE;



    GROW_ARRAY(fg->inputs, fg->nb_inputs);

    if (!(fg->inputs[fg->nb_inputs - 1] = av_mallocz(sizeof(*fg->inputs[0]))))

        exit(1);

    fg->inputs[fg->nb_inputs - 1]->ist   = ist;

    fg->inputs[fg->nb_inputs - 1]->graph = fg;

    fg->inputs[fg->nb_inputs - 1]->format = -1;



    fg->inputs[fg->nb_inputs - 1]->frame_queue = av_fifo_alloc(8 * sizeof(AVFrame*));

    if (!fg->inputs[fg->nb_inputs - 1])

        exit_program(1);



    GROW_ARRAY(ist->filters, ist->nb_filters);

    ist->filters[ist->nb_filters - 1] = fg->inputs[fg->nb_inputs - 1];

}
