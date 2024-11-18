AVStream *add_audio_stream(AVFormatContext *oc, int codec_id)

{

    AVCodec *codec;

    AVCodecContext *c;

    AVStream *st;



    st = av_new_stream(oc, 1);

    if (!st) {

        fprintf(stderr, "Could not alloc stream\n");

        exit(1);

    }



    /* find the MP2 encoder */

    codec = avcodec_find_encoder(codec_id);

    if (!codec) {

        fprintf(stderr, "codec not found\n");

        exit(1);

    }

    c = &st->codec;

    c->codec_type = CODEC_TYPE_AUDIO;



    /* put sample parameters */

    c->bit_rate = 64000;

    c->sample_rate = 44100;

    c->channels = 2;



    /* open it */

    if (avcodec_open(c, codec) < 0) {

        fprintf(stderr, "could not open codec\n");

        exit(1);

    }



    /* init signal generator */

    t = 0;

    tincr = 2 * M_PI * 440.0 / c->sample_rate;



    audio_outbuf_size = 10000;

    audio_outbuf = malloc(audio_outbuf_size);



    /* ugly hack for PCM codecs (will be removed ASAP with new PCM

       support to compute the input frame size in samples */

    if (c->frame_size <= 1) {

        audio_input_frame_size = audio_outbuf_size / c->channels;

        switch(st->codec.codec_id) {

        case CODEC_ID_PCM_S16LE:

        case CODEC_ID_PCM_S16BE:

        case CODEC_ID_PCM_U16LE:

        case CODEC_ID_PCM_U16BE:

            audio_input_frame_size >>= 1;

            break;

        default:

            break;

        }

    } else {

        audio_input_frame_size = c->frame_size;

    }

    samples = malloc(audio_input_frame_size * 2 * c->channels);



    return st;

}
