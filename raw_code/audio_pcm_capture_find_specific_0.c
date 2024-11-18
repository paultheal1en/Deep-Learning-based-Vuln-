static CaptureVoiceOut *audio_pcm_capture_find_specific (

    AudioState *s,

    audsettings_t *as

    )

{

    CaptureVoiceOut *cap;



    for (cap = s->cap_head.lh_first; cap; cap = cap->entries.le_next) {

        if (audio_pcm_info_eq (&cap->hw.info, as)) {

            return cap;

        }

    }

    return NULL;

}
