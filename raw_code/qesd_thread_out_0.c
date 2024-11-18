static void *qesd_thread_out (void *arg)

{

    ESDVoiceOut *esd = arg;

    HWVoiceOut *hw = &esd->hw;

    int threshold;



    threshold = conf.divisor ? hw->samples / conf.divisor : 0;



    if (audio_pt_lock (&esd->pt, AUDIO_FUNC)) {

        return NULL;

    }



    for (;;) {

        int decr, to_mix, rpos;



        for (;;) {

            if (esd->done) {

                goto exit;

            }



            if (esd->live > threshold) {

                break;

            }



            if (audio_pt_wait (&esd->pt, AUDIO_FUNC)) {

                goto exit;

            }

        }



        decr = to_mix = esd->live;

        rpos = hw->rpos;



        if (audio_pt_unlock (&esd->pt, AUDIO_FUNC)) {

            return NULL;

        }



        while (to_mix) {

            ssize_t written;

            int chunk = audio_MIN (to_mix, hw->samples - rpos);

            st_sample_t *src = hw->mix_buf + rpos;



            hw->clip (esd->pcm_buf, src, chunk);



        again:

            written = write (esd->fd, esd->pcm_buf, chunk << hw->info.shift);

            if (written == -1) {

                if (errno == EINTR || errno == EAGAIN) {

                    goto again;

                }

                qesd_logerr (errno, "write failed\n");

                return NULL;

            }



            if (written != chunk << hw->info.shift) {

                int wsamples = written >> hw->info.shift;

                int wbytes = wsamples << hw->info.shift;

                if (wbytes != written) {

                    dolog ("warning: Misaligned write %d (requested %d), "

                           "alignment %d\n",

                           wbytes, written, hw->info.align + 1);

                }

                to_mix -= wsamples;

                rpos = (rpos + wsamples) % hw->samples;

                break;

            }



            rpos = (rpos + chunk) % hw->samples;

            to_mix -= chunk;

        }



        if (audio_pt_lock (&esd->pt, AUDIO_FUNC)) {

            return NULL;

        }



        esd->rpos = rpos;

        esd->live -= decr;

        esd->decr += decr;

    }



 exit:

    audio_pt_unlock (&esd->pt, AUDIO_FUNC);

    return NULL;

}
