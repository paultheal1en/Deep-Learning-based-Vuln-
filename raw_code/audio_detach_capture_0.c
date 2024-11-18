static void audio_detach_capture (HWVoiceOut *hw)

{

    SWVoiceCap *sc = hw->cap_head.lh_first;



    while (sc) {

        SWVoiceCap *sc1 = sc->entries.le_next;

        SWVoiceOut *sw = &sc->sw;

        CaptureVoiceOut *cap = sc->cap;

        int was_active = sw->active;



        if (sw->rate) {

            st_rate_stop (sw->rate);

            sw->rate = NULL;

        }



        LIST_REMOVE (sw, entries);

        LIST_REMOVE (sc, entries);

        qemu_free (sc);

        if (was_active) {

            /* We have removed soft voice from the capture:

               this might have changed the overall status of the capture

               since this might have been the only active voice */

            audio_recalc_and_notify_capture (cap);

        }

        sc = sc1;

    }

}
