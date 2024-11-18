static double get_audio_clock(VideoState *is)

{

    double pts;

    int hw_buf_size, bytes_per_sec;

    pts = is->audio_clock;

    hw_buf_size = audio_write_get_buf_size(is);

    bytes_per_sec = 0;

    if (is->audio_st) {

        bytes_per_sec = is->audio_st->codec->sample_rate *

                        2 * is->audio_st->codec->channels;

    }

    if (bytes_per_sec)

        pts -= (double)hw_buf_size / bytes_per_sec;

    return pts;

}
