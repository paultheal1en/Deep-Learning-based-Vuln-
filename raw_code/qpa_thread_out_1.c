static void *qpa_thread_out (void *arg)

{

    PAVoiceOut *pa = arg;

    HWVoiceOut *hw = &pa->hw;

    int threshold;



    threshold = conf.divisor ? hw->samples / conf.divisor : 0;



    if (audio_pt_lock (&pa->pt, AUDIO_FUNC)) {

        return NULL;

    }



    for (;;) {

        int decr, to_mix, rpos;



        for (;;) {

            if (pa->done) {

                goto exit;

            }



            if (pa->live > threshold) {

                break;

            }



            if (audio_pt_wait (&pa->pt, AUDIO_FUNC)) {

                goto exit;

            }

        }



        decr = to_mix = pa->live;

        rpos = hw->rpos;



        if (audio_pt_unlock (&pa->pt, AUDIO_FUNC)) {

            return NULL;

        }



        while (to_mix) {

            int error;

            int chunk = audio_MIN (to_mix, hw->samples - rpos);

            struct st_sample *src = hw->mix_buf + rpos;



            hw->clip (pa->pcm_buf, src, chunk);



            if (pa_simple_write (pa->s, pa->pcm_buf,

                                 chunk << hw->info.shift, &error) < 0) {

                qpa_logerr (error, "pa_simple_write failed\n");

                return NULL;

            }



            rpos = (rpos + chunk) % hw->samples;

            to_mix -= chunk;

        }



        if (audio_pt_lock (&pa->pt, AUDIO_FUNC)) {

            return NULL;

        }



        pa->rpos = rpos;

        pa->live -= decr;

        pa->decr += decr;

    }



 exit:

    audio_pt_unlock (&pa->pt, AUDIO_FUNC);

    return NULL;

}
