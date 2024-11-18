static int init_output_stream(OutputStream *ost, char *error, int error_len)

{

    int ret = 0;



    if (ost->encoding_needed) {

        AVCodec      *codec = ost->enc;

        AVCodecContext *dec = NULL;

        InputStream *ist;



        ret = init_output_stream_encode(ost);

        if (ret < 0)

            return ret;



        if ((ist = get_input_stream(ost)))

            dec = ist->dec_ctx;

        if (dec && dec->subtitle_header) {

            ost->enc_ctx->subtitle_header = av_malloc(dec->subtitle_header_size);

            if (!ost->enc_ctx->subtitle_header)

                return AVERROR(ENOMEM);

            memcpy(ost->enc_ctx->subtitle_header, dec->subtitle_header, dec->subtitle_header_size);

            ost->enc_ctx->subtitle_header_size = dec->subtitle_header_size;

        }

        if (!av_dict_get(ost->encoder_opts, "threads", NULL, 0))

            av_dict_set(&ost->encoder_opts, "threads", "auto", 0);



        if (ost->filter && ost->filter->filter->inputs[0]->hw_frames_ctx) {

            ost->enc_ctx->hw_frames_ctx = av_buffer_ref(ost->filter->filter->inputs[0]->hw_frames_ctx);

            if (!ost->enc_ctx->hw_frames_ctx)

                return AVERROR(ENOMEM);

        }



        if ((ret = avcodec_open2(ost->enc_ctx, codec, &ost->encoder_opts)) < 0) {

            if (ret == AVERROR_EXPERIMENTAL)

                abort_codec_experimental(codec, 1);

            snprintf(error, error_len,

                     "Error while opening encoder for output stream #%d:%d - "

                     "maybe incorrect parameters such as bit_rate, rate, width or height",

                    ost->file_index, ost->index);

            return ret;

        }

        assert_avoptions(ost->encoder_opts);

        if (ost->enc_ctx->bit_rate && ost->enc_ctx->bit_rate < 1000)

            av_log(NULL, AV_LOG_WARNING, "The bitrate parameter is set too low."

                                         "It takes bits/s as argument, not kbits/s\n");



        ret = avcodec_parameters_from_context(ost->st->codecpar, ost->enc_ctx);

        if (ret < 0) {

            av_log(NULL, AV_LOG_FATAL,

                   "Error initializing the output stream codec context.\n");

            exit_program(1);

        }



        if (ost->enc_ctx->nb_coded_side_data) {

            int i;



            ost->st->side_data = av_realloc_array(NULL, ost->enc_ctx->nb_coded_side_data,

                                                  sizeof(*ost->st->side_data));

            if (!ost->st->side_data)

                return AVERROR(ENOMEM);



            for (i = 0; i < ost->enc_ctx->nb_coded_side_data; i++) {

                const AVPacketSideData *sd_src = &ost->enc_ctx->coded_side_data[i];

                AVPacketSideData *sd_dst = &ost->st->side_data[i];



                sd_dst->data = av_malloc(sd_src->size);

                if (!sd_dst->data)

                    return AVERROR(ENOMEM);

                memcpy(sd_dst->data, sd_src->data, sd_src->size);

                sd_dst->size = sd_src->size;

                sd_dst->type = sd_src->type;

                ost->st->nb_side_data++;

            }

        }



        ost->st->time_base = ost->enc_ctx->time_base;

    } else if (ost->stream_copy) {

        ret = init_output_stream_streamcopy(ost);

        if (ret < 0)

            return ret;



        /*

         * FIXME: will the codec context used by the parser during streamcopy

         * This should go away with the new parser API.

         */

        ret = avcodec_parameters_to_context(ost->parser_avctx, ost->st->codecpar);

        if (ret < 0)

            return ret;

    }



    /* initialize bitstream filters for the output stream

     * needs to be done here, because the codec id for streamcopy is not

     * known until now */

    ret = init_output_bsfs(ost);

    if (ret < 0)

        return ret;



    ost->mux_timebase = ost->st->time_base;



    ost->initialized = 1;



    ret = check_init_output_file(output_files[ost->file_index], ost->file_index);

    if (ret < 0)

        return ret;



    return ret;

}
