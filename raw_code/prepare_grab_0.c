void prepare_grab(void)

{

    int has_video, has_audio, i, j;

    AVFormatContext *oc;

    AVFormatContext *ic;

    AVFormatParameters ap1, *ap = &ap1;



    /* see if audio/video inputs are needed */

    has_video = 0;

    has_audio = 0;

    memset(ap, 0, sizeof(*ap));

    for(j=0;j<nb_output_files;j++) {

        oc = output_files[j];

        for(i=0;i<oc->nb_streams;i++) {

            AVCodecContext *enc = &oc->streams[i]->codec;

            switch(enc->codec_type) {

            case CODEC_TYPE_AUDIO:

                if (enc->sample_rate > ap->sample_rate)

                    ap->sample_rate = enc->sample_rate;

                if (enc->channels > ap->channels)

                    ap->channels = enc->channels;

                has_audio = 1;

                break;

            case CODEC_TYPE_VIDEO:

                if (enc->width > ap->width)

                    ap->width = enc->width;

                if (enc->height > ap->height)

                    ap->height = enc->height;

                if (enc->frame_rate > ap->frame_rate)

                    ap->frame_rate = enc->frame_rate;

                has_video = 1;

                break;

            default:

                abort();

            }

        }

    }

    

    if (has_video == 0 && has_audio == 0) {

        fprintf(stderr, "Output file must have at least one audio or video stream\n");

        exit(1);

    }

    

    if (has_video) {

        AVInputFormat *fmt1;

        fmt1 = av_find_input_format("video_grab_device");

        if (av_open_input_file(&ic, "", fmt1, 0, ap) < 0) {

            fprintf(stderr, "Could not find video grab device\n");

            exit(1);

        }

        /* by now video grab has one stream */

        ic->streams[0]->r_frame_rate = ap->frame_rate;

        input_files[nb_input_files] = ic;

        dump_format(ic, nb_input_files, v4l_device, 0);

        nb_input_files++;

    }

    if (has_audio) {

        AVInputFormat *fmt1;

        fmt1 = av_find_input_format("audio_device");

        if (av_open_input_file(&ic, "", fmt1, 0, ap) < 0) {

            fprintf(stderr, "Could not find audio grab device\n");

            exit(1);

        }

        input_files[nb_input_files] = ic;

        dump_format(ic, nb_input_files, audio_device, 0);

        nb_input_files++;

    }

}
