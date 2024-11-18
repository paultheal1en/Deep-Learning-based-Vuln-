static void audio_run_capture (AudioState *s)

{

    CaptureVoiceOut *cap;



    for (cap = s->cap_head.lh_first; cap; cap = cap->entries.le_next) {

        int live, rpos, captured;

        HWVoiceOut *hw = &cap->hw;

        SWVoiceOut *sw;



        captured = live = audio_pcm_hw_get_live_out (hw);

        rpos = hw->rpos;

        while (live) {

            int left = hw->samples - rpos;

            int to_capture = audio_MIN (live, left);

            st_sample_t *src;

            struct capture_callback *cb;



            src = hw->mix_buf + rpos;

            hw->clip (cap->buf, src, to_capture);

            mixeng_clear (src, to_capture);



            for (cb = cap->cb_head.lh_first; cb; cb = cb->entries.le_next) {

                cb->ops.capture (cb->opaque, cap->buf,

                                 to_capture << hw->info.shift);

            }

            rpos = (rpos + to_capture) % hw->samples;

            live -= to_capture;

        }

        hw->rpos = rpos;



        for (sw = hw->sw_head.lh_first; sw; sw = sw->entries.le_next) {

            if (!sw->active && sw->empty) {

                continue;

            }



            if (audio_bug (AUDIO_FUNC, captured > sw->total_hw_samples_mixed)) {

                dolog ("captured=%d sw->total_hw_samples_mixed=%d\n",

                       captured, sw->total_hw_samples_mixed);

                captured = sw->total_hw_samples_mixed;

            }



            sw->total_hw_samples_mixed -= captured;

            sw->empty = sw->total_hw_samples_mixed == 0;

        }

    }

}
