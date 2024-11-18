void opt_output_file(const char *filename)

{

    AVStream *st;

    AVFormatContext *oc;

    int use_video, use_audio, nb_streams, input_has_video, input_has_audio;

    int codec_id;



    if (!strcmp(filename, "-"))

        filename = "pipe:";



    oc = av_mallocz(sizeof(AVFormatContext));



    if (!file_format) {

        file_format = guess_format(NULL, filename, NULL);

        if (!file_format)

            file_format = &mpeg_mux_format;

    }

    

    oc->format = file_format;



    if (!strcmp(file_format->name, "ffm") && 

        strstart(filename, "http:", NULL)) {

        /* special case for files sent to ffserver: we get the stream

           parameters from ffserver */

        if (read_ffserver_streams(oc, filename) < 0) {

            fprintf(stderr, "Could not read stream parameters from '%s'\n", filename);

            exit(1);

        }

    } else {

        use_video = file_format->video_codec != CODEC_ID_NONE;

        use_audio = file_format->audio_codec != CODEC_ID_NONE;



        /* disable if no corresponding type found */

        check_audio_video_inputs(&input_has_video, &input_has_audio);

        if (!input_has_video)

            use_video = 0;

        if (!input_has_audio)

            use_audio = 0;



        /* manual disable */

        if (audio_disable) {

            use_audio = 0;

        }

        if (video_disable) {

            use_video = 0;

        }

        

        nb_streams = 0;

        if (use_video) {

            AVCodecContext *video_enc;

            

            st = av_mallocz(sizeof(AVStream));

            if (!st) {

                fprintf(stderr, "Could not alloc stream\n");

                exit(1);

            }

            video_enc = &st->codec;



            codec_id = file_format->video_codec;

            if (video_codec_id != CODEC_ID_NONE)

                codec_id = video_codec_id;



            video_enc->codec_id = codec_id;

            video_enc->codec_type = CODEC_TYPE_VIDEO;

            

            video_enc->bit_rate = video_bit_rate;

            video_enc->frame_rate = frame_rate; 

            

            video_enc->width = frame_width;

            video_enc->height = frame_height;

            if (!intra_only)

                video_enc->gop_size = gop_size;

            else

                video_enc->gop_size = 0;

            if (video_qscale || same_quality) {

                video_enc->flags |= CODEC_FLAG_QSCALE;

                video_enc->quality = video_qscale;

            }

            /* XXX: need to find a way to set codec parameters */

            if (oc->format == &ppm_format ||

                oc->format == &ppmpipe_format) {

                video_enc->pix_fmt = PIX_FMT_RGB24;

            }



            oc->streams[nb_streams] = st;

            nb_streams++;

        }

    

        if (use_audio) {

            AVCodecContext *audio_enc;



            st = av_mallocz(sizeof(AVStream));

            if (!st) {

                fprintf(stderr, "Could not alloc stream\n");

                exit(1);

            }

            audio_enc = &st->codec;

            codec_id = file_format->audio_codec;

            if (audio_codec_id != CODEC_ID_NONE)

                codec_id = audio_codec_id;

            audio_enc->codec_id = codec_id;

            audio_enc->codec_type = CODEC_TYPE_AUDIO;

            

            audio_enc->bit_rate = audio_bit_rate;

            audio_enc->sample_rate = audio_sample_rate;

            audio_enc->channels = audio_channels;

            oc->streams[nb_streams] = st;

            nb_streams++;

        }



        oc->nb_streams = nb_streams;



        if (!nb_streams) {

            fprintf(stderr, "No audio or video streams available\n");

            exit(1);

        }



        if (str_title)

            nstrcpy(oc->title, sizeof(oc->title), str_title);

        if (str_author)

            nstrcpy(oc->author, sizeof(oc->author), str_author);

        if (str_copyright)

            nstrcpy(oc->copyright, sizeof(oc->copyright), str_copyright);

        if (str_comment)

            nstrcpy(oc->comment, sizeof(oc->comment), str_comment);

    }



    output_files[nb_output_files] = oc;

    /* dump the file content */

    dump_format(oc, nb_output_files, filename, 1);

    nb_output_files++;



    strcpy(oc->filename, filename);



    /* check filename in case of an image number is expected */

    if (oc->format->flags & AVFMT_NEEDNUMBER) {

        if (filename_number_test(oc->filename) < 0)

            exit(1);

    }



    if (!(oc->format->flags & AVFMT_NOFILE)) {

        /* test if it already exists to avoid loosing precious files */

        if (!file_overwrite && 

            (strchr(filename, ':') == NULL ||

             strstart(filename, "file:", NULL))) {

            if (url_exist(filename)) {

                int c;

                

                printf("File '%s' already exists. Overwrite ? [y/N] ", filename);

                fflush(stdout);

                c = getchar();

                if (toupper(c) != 'Y') {

                    fprintf(stderr, "Not overwriting - exiting\n");

                    exit(1);

                }

            }

        }

        

        /* open the file */

        if (url_fopen(&oc->pb, filename, URL_WRONLY) < 0) {

            fprintf(stderr, "Could not open '%s'\n", filename);

            exit(1);

        }

    }



    /* reset some options */

    file_format = NULL;

    audio_disable = 0;

    video_disable = 0;

    audio_codec_id = CODEC_ID_NONE;

    video_codec_id = CODEC_ID_NONE;

}
