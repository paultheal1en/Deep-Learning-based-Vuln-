void opt_input_file(const char *filename)

{

    AVFormatContext *ic;

    AVFormatParameters params, *ap = &params;

    int err, i, ret, rfps;



    /* get default parameters from command line */

    memset(ap, 0, sizeof(*ap));

    ap->sample_rate = audio_sample_rate;

    ap->channels = audio_channels;

    ap->frame_rate = frame_rate;

    ap->width = frame_width;

    ap->height = frame_height;



    /* open the input file with generic libav function */

    err = av_open_input_file(&ic, filename, file_iformat, 0, ap);

    if (err < 0) {

        print_error(filename, err);

        exit(1);

    }

    

    /* If not enough info to get the stream parameters, we decode the

       first frames to get it. (used in mpeg case for example) */

    ret = av_find_stream_info(ic);

    if (ret < 0) {

        fprintf(stderr, "%s: could not find codec parameters\n", filename);

        exit(1);

    }



    /* update the current parameters so that they match the one of the input stream */

    for(i=0;i<ic->nb_streams;i++) {

        AVCodecContext *enc = &ic->streams[i]->codec;

        switch(enc->codec_type) {

        case CODEC_TYPE_AUDIO:

            //fprintf(stderr, "\nInput Audio channels: %d", enc->channels);

            audio_channels = enc->channels;

            audio_sample_rate = enc->sample_rate;

            break;

        case CODEC_TYPE_VIDEO:

            frame_height = enc->height;

            frame_width = enc->width;

            rfps = ic->streams[i]->r_frame_rate;

            if (enc->frame_rate != rfps) {

                fprintf(stderr,"\nSeems that stream %d comes from film source: %2.2f->%2.2f\n",

                    i, (float)enc->frame_rate / FRAME_RATE_BASE,

                    (float)rfps / FRAME_RATE_BASE);

            }

            /* update the current frame rate to match the stream frame rate */

            frame_rate = rfps;

            break;

        default:

            abort();

        }

    }

    

    input_files[nb_input_files] = ic;

    /* dump the file content */

    dump_format(ic, nb_input_files, filename, 0);

    nb_input_files++;

    file_iformat = NULL;

    file_oformat = NULL;

}
