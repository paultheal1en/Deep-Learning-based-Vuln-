static void hda_audio_exit(HDACodecDevice *hda)

{

    HDAAudioState *a = HDA_AUDIO(hda);

    HDAAudioStream *st;

    int i;



    dprint(a, 1, "%s\n", __FUNCTION__);

    for (i = 0; i < ARRAY_SIZE(a->st); i++) {

        st = a->st + i;

        if (st->node == NULL) {

            continue;

        }

        if (st->output) {

            AUD_close_out(&a->card, st->voice.out);

        } else {

            AUD_close_in(&a->card, st->voice.in);

        }

    }

    AUD_remove_card(&a->card);

}
