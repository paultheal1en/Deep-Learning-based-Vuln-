static av_cold int audio_write_header(AVFormatContext *s1)

{

    AlsaData *s = s1->priv_data;

    AVStream *st;

    unsigned int sample_rate;

    enum AVCodecID codec_id;

    int res;



    st = s1->streams[0];

    sample_rate = st->codec->sample_rate;

    codec_id    = st->codec->codec_id;

    res = ff_alsa_open(s1, SND_PCM_STREAM_PLAYBACK, &sample_rate,

        st->codec->channels, &codec_id);

    if (sample_rate != st->codec->sample_rate) {

        av_log(s1, AV_LOG_ERROR,

               "sample rate %d not available, nearest is %d\n",

               st->codec->sample_rate, sample_rate);

        goto fail;

    }

    avpriv_set_pts_info(st, 64, 1, sample_rate);



    return res;



fail:

    snd_pcm_close(s->h);

    return AVERROR(EIO);

}
