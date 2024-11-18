int audio_pcm_sw_read (SWVoiceIn *sw, void *buf, int size)

{

    HWVoiceIn *hw = sw->hw;

    int samples, live, ret = 0, swlim, isamp, osamp, rpos, total = 0;

    st_sample_t *src, *dst = sw->buf;



    rpos = audio_pcm_sw_get_rpos_in (sw) % hw->samples;



    live = hw->total_samples_captured - sw->total_hw_samples_acquired;

    if (audio_bug (AUDIO_FUNC, live < 0 || live > hw->samples)) {

        dolog ("live_in=%d hw->samples=%d\n", live, hw->samples);

        return 0;

    }



    samples = size >> sw->info.shift;

    if (!live) {

        return 0;

    }



    swlim = (live * sw->ratio) >> 32;

    swlim = audio_MIN (swlim, samples);



    while (swlim) {

        src = hw->conv_buf + rpos;

        isamp = hw->wpos - rpos;

        /* XXX: <= ? */

        if (isamp <= 0) {

            isamp = hw->samples - rpos;

        }



        if (!isamp) {

            break;

        }

        osamp = swlim;



        if (audio_bug (AUDIO_FUNC, osamp < 0)) {

            dolog ("osamp=%d\n", osamp);

            return 0;

        }



        st_rate_flow (sw->rate, src, dst, &isamp, &osamp);

        swlim -= osamp;

        rpos = (rpos + isamp) % hw->samples;

        dst += osamp;

        ret += osamp;

        total += isamp;

    }



    sw->clip (buf, sw->buf, ret);

    sw->total_hw_samples_acquired += total;

    return ret << sw->info.shift;

}
