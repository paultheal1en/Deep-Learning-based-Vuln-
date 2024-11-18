static void sdl_audio_callback(void *opaque, Uint8 *stream, int len)

{

    VideoState *is = opaque;

    int audio_size, len1;

    int bytes_per_sec;

    int frame_size = av_samples_get_buffer_size(NULL, is->audio_tgt.channels, 1, is->audio_tgt.fmt, 1);

    double pts;



    audio_callback_time = av_gettime();



    while (len > 0) {

        if (is->audio_buf_index >= is->audio_buf_size) {

           audio_size = audio_decode_frame(is, &pts);

           if (audio_size < 0) {

                /* if error, just output silence */

               is->audio_buf      = is->silence_buf;

               is->audio_buf_size = sizeof(is->silence_buf) / frame_size * frame_size;

           } else {

               if (is->show_mode != SHOW_MODE_VIDEO)

                   update_sample_display(is, (int16_t *)is->audio_buf, audio_size);

               is->audio_buf_size = audio_size;

           }

           is->audio_buf_index = 0;

        }

        len1 = is->audio_buf_size - is->audio_buf_index;

        if (len1 > len)

            len1 = len;

        memcpy(stream, (uint8_t *)is->audio_buf + is->audio_buf_index, len1);

        len -= len1;

        stream += len1;

        is->audio_buf_index += len1;

    }

    bytes_per_sec = is->audio_tgt.freq * is->audio_tgt.channels * av_get_bytes_per_sample(is->audio_tgt.fmt);

    is->audio_write_buf_size = is->audio_buf_size - is->audio_buf_index;

    /* Let's assume the audio driver that is used by SDL has two periods. */

    is->audio_current_pts = is->audio_clock - (double)(2 * is->audio_hw_buf_size + is->audio_write_buf_size) / bytes_per_sec;

    is->audio_current_pts_drift = is->audio_current_pts - audio_callback_time / 1000000.0;

    check_external_clock_sync(is, is->audio_current_pts);

}
