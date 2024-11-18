static av_cold int pulse_write_header(AVFormatContext *h)

{

    PulseData *s = h->priv_data;

    AVStream *st = NULL;

    int ret;

    unsigned int i;

    pa_sample_spec ss;

    pa_buffer_attr attr = { -1, -1, -1, -1, -1 };

    const char *stream_name = s->stream_name;



    for (i = 0; i < h->nb_streams; i++) {

        if (h->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            st = h->streams[i];

            s->stream_index = i;

            break;

        }

    }



    if (!st) {

        av_log(s, AV_LOG_ERROR, "No audio stream found.\n");

        return AVERROR(EINVAL);

    }



    if (!stream_name) {

        if (h->filename[0])

            stream_name = h->filename;

        else

            stream_name = "Playback";

    }



    ss.format = codec_id_to_pulse_format(st->codec->codec_id);

    ss.rate = st->codec->sample_rate;

    ss.channels = st->codec->channels;



    s->pa = pa_simple_new(s->server,                 // Server

                          s->name,                   // Application name

                          PA_STREAM_PLAYBACK,

                          s->device,                 // Device

                          stream_name,               // Description of a stream

                          &ss,                       // Sample format

                          NULL,                      // Use default channel map

                          &attr,                     // Buffering attributes

                          &ret);                     // Result



    if (!s->pa) {

        av_log(s, AV_LOG_ERROR, "pa_simple_new failed: %s\n", pa_strerror(ret));

        return AVERROR(EIO);

    }



    avpriv_set_pts_info(st, 64, 1, 1000000);  /* 64 bits pts in us */



    return 0;

}
