static int oss_run_out (HWVoiceOut *hw)

{

    OSSVoiceOut *oss = (OSSVoiceOut *) hw;

    int err, rpos, live, decr;

    int samples;

    uint8_t *dst;

    st_sample_t *src;

    struct audio_buf_info abinfo;

    struct count_info cntinfo;

    int bufsize;



    live = audio_pcm_hw_get_live_out (hw);

    if (!live) {

        return 0;

    }



    bufsize = hw->samples << hw->info.shift;



    if (oss->mmapped) {

        int bytes;



        err = ioctl (oss->fd, SNDCTL_DSP_GETOPTR, &cntinfo);

        if (err < 0) {

            oss_logerr (errno, "SNDCTL_DSP_GETOPTR failed\n");

            return 0;

        }



        if (cntinfo.ptr == oss->old_optr) {

            if (abs (hw->samples - live) < 64) {

                dolog ("warning: Overrun\n");

            }

            return 0;

        }



        if (cntinfo.ptr > oss->old_optr) {

            bytes = cntinfo.ptr - oss->old_optr;

        }

        else {

            bytes = bufsize + cntinfo.ptr - oss->old_optr;

        }



        decr = audio_MIN (bytes >> hw->info.shift, live);

    }

    else {

        err = ioctl (oss->fd, SNDCTL_DSP_GETOSPACE, &abinfo);

        if (err < 0) {

            oss_logerr (errno, "SNDCTL_DSP_GETOPTR failed\n");

            return 0;

        }



        if (abinfo.bytes > bufsize) {

            if (conf.debug) {

                dolog ("warning: Invalid available size, size=%d bufsize=%d\n"

                       "please report your OS/audio hw to malc@pulsesoft.com\n",

                       abinfo.bytes, bufsize);

            }

            abinfo.bytes = bufsize;

        }



        if (abinfo.bytes < 0) {

            if (conf.debug) {

                dolog ("warning: Invalid available size, size=%d bufsize=%d\n",

                       abinfo.bytes, bufsize);

            }

            return 0;

        }



        decr = audio_MIN (abinfo.bytes >> hw->info.shift, live);

        if (!decr) {

            return 0;

        }

    }



    samples = decr;

    rpos = hw->rpos;

    while (samples) {

        int left_till_end_samples = hw->samples - rpos;

        int convert_samples = audio_MIN (samples, left_till_end_samples);



        src = hw->mix_buf + rpos;

        dst = advance (oss->pcm_buf, rpos << hw->info.shift);



        hw->clip (dst, src, convert_samples);

        if (!oss->mmapped) {

            int written;



            written = write (oss->fd, dst, convert_samples << hw->info.shift);

            /* XXX: follow errno recommendations ? */

            if (written == -1) {

                oss_logerr (

                    errno,

                    "Failed to write %d bytes of audio data from %p\n",

                    convert_samples << hw->info.shift,

                    dst

                    );

                continue;

            }



            if (written != convert_samples << hw->info.shift) {

                int wsamples = written >> hw->info.shift;

                int wbytes = wsamples << hw->info.shift;

                if (wbytes != written) {

                    dolog ("warning: Misaligned write %d (requested %d), "

                           "alignment %d\n",

                           wbytes, written, hw->info.align + 1);

                }

                decr -= wsamples;

                rpos = (rpos + wsamples) % hw->samples;

                break;

            }

        }



        rpos = (rpos + convert_samples) % hw->samples;

        samples -= convert_samples;

    }

    if (oss->mmapped) {

        oss->old_optr = cntinfo.ptr;

    }



    hw->rpos = rpos;

    return decr;

}
