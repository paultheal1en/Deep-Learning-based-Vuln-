static int copy_metadata(char *outspec, char *inspec, AVFormatContext *oc, AVFormatContext *ic, OptionsContext *o)

{

    AVDictionary **meta_in = NULL;

    AVDictionary **meta_out;

    int i, ret = 0;

    char type_in, type_out;

    const char *istream_spec = NULL, *ostream_spec = NULL;

    int idx_in = 0, idx_out = 0;



    parse_meta_type(inspec,  &type_in,  &idx_in,  &istream_spec);

    parse_meta_type(outspec, &type_out, &idx_out, &ostream_spec);



    if (type_in == 'g' || type_out == 'g')

        o->metadata_global_manual = 1;

    if (type_in == 's' || type_out == 's')

        o->metadata_streams_manual = 1;

    if (type_in == 'c' || type_out == 'c')

        o->metadata_chapters_manual = 1;



#define METADATA_CHECK_INDEX(index, nb_elems, desc)\

    if ((index) < 0 || (index) >= (nb_elems)) {\

        av_log(NULL, AV_LOG_FATAL, "Invalid %s index %d while processing metadata maps.\n",\

                (desc), (index));\

        exit_program(1);\

    }



#define SET_DICT(type, meta, context, index)\

        switch (type) {\

        case 'g':\

            meta = &context->metadata;\

            break;\

        case 'c':\

            METADATA_CHECK_INDEX(index, context->nb_chapters, "chapter")\

            meta = &context->chapters[index]->metadata;\

            break;\

        case 'p':\

            METADATA_CHECK_INDEX(index, context->nb_programs, "program")\

            meta = &context->programs[index]->metadata;\

            break;\


        }\



    SET_DICT(type_in, meta_in, ic, idx_in);

    SET_DICT(type_out, meta_out, oc, idx_out);



    /* for input streams choose first matching stream */

    if (type_in == 's') {

        for (i = 0; i < ic->nb_streams; i++) {

            if ((ret = check_stream_specifier(ic, ic->streams[i], istream_spec)) > 0) {

                meta_in = &ic->streams[i]->metadata;

                break;

            } else if (ret < 0)

                exit_program(1);

        }

        if (!meta_in) {

            av_log(NULL, AV_LOG_FATAL, "Stream specifier %s does not match  any streams.\n", istream_spec);

            exit_program(1);

        }

    }



    if (type_out == 's') {

        for (i = 0; i < oc->nb_streams; i++) {

            if ((ret = check_stream_specifier(oc, oc->streams[i], ostream_spec)) > 0) {

                meta_out = &oc->streams[i]->metadata;

                av_dict_copy(meta_out, *meta_in, AV_DICT_DONT_OVERWRITE);

            } else if (ret < 0)

                exit_program(1);

        }

    } else

        av_dict_copy(meta_out, *meta_in, AV_DICT_DONT_OVERWRITE);



    return 0;

}