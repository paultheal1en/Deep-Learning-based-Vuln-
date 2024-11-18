static int sls_flag_use_localtime_filename(AVFormatContext *oc, HLSContext *c, VariantStream *vs)

{

    if (c->flags & HLS_SECOND_LEVEL_SEGMENT_INDEX) {

        char * filename = av_strdup(oc->filename);  // %%d will be %d after strftime

        if (!filename)

            return AVERROR(ENOMEM);

        if (replace_int_data_in_filename(oc->filename, sizeof(oc->filename),

#if FF_API_HLS_WRAP

            filename, 'd', c->wrap ? vs->sequence % c->wrap : vs->sequence) < 1) {

#else

            filename, 'd', vs->sequence) < 1) {

#endif

            av_log(c, AV_LOG_ERROR, "Invalid second level segment filename template '%s', "

                    "you can try to remove second_level_segment_index flag\n",

                   filename);

            av_free(filename);

            return AVERROR(EINVAL);

        }

        av_free(filename);

    }

    if (c->flags & (HLS_SECOND_LEVEL_SEGMENT_SIZE | HLS_SECOND_LEVEL_SEGMENT_DURATION)) {

        av_strlcpy(vs->current_segment_final_filename_fmt, oc->filename,

                   sizeof(vs->current_segment_final_filename_fmt));

        if (c->flags & HLS_SECOND_LEVEL_SEGMENT_SIZE) {

            char * filename = av_strdup(oc->filename);  // %%s will be %s after strftime

            if (!filename)

                return AVERROR(ENOMEM);

            if (replace_int_data_in_filename(oc->filename, sizeof(oc->filename), filename, 's', 0) < 1) {

                av_log(c, AV_LOG_ERROR, "Invalid second level segment filename template '%s', "

                        "you can try to remove second_level_segment_size flag\n",

                       filename);

                av_free(filename);

                return AVERROR(EINVAL);

            }

            av_free(filename);

        }

        if (c->flags & HLS_SECOND_LEVEL_SEGMENT_DURATION) {

            char * filename = av_strdup(oc->filename);  // %%t will be %t after strftime

            if (!filename)

                return AVERROR(ENOMEM);

            if (replace_int_data_in_filename(oc->filename, sizeof(oc->filename), filename, 't', 0) < 1) {

                av_log(c, AV_LOG_ERROR, "Invalid second level segment filename template '%s', "

                        "you can try to remove second_level_segment_time flag\n",

                       filename);

                av_free(filename);

                return AVERROR(EINVAL);

            }

            av_free(filename);

        }

    }

    return 0;

}
