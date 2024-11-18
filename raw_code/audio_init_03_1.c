static void audio_init (void)

{

    size_t i;

    int done = 0;

    const char *drvname;

    VMChangeStateEntry *e;

    AudioState *s = &glob_audio_state;



    if (s->drv) {

        return;

    }



    QLIST_INIT (&s->hw_head_out);

    QLIST_INIT (&s->hw_head_in);

    QLIST_INIT (&s->cap_head);

    atexit (audio_atexit);



    s->ts = timer_new_ns(QEMU_CLOCK_VIRTUAL, audio_timer, s);

    if (!s->ts) {

        hw_error("Could not create audio timer\n");

    }



    audio_process_options ("AUDIO", audio_options);



    s->nb_hw_voices_out = conf.fixed_out.nb_voices;

    s->nb_hw_voices_in = conf.fixed_in.nb_voices;



    if (s->nb_hw_voices_out <= 0) {

        dolog ("Bogus number of playback voices %d, setting to 1\n",

               s->nb_hw_voices_out);

        s->nb_hw_voices_out = 1;

    }



    if (s->nb_hw_voices_in <= 0) {

        dolog ("Bogus number of capture voices %d, setting to 0\n",

               s->nb_hw_voices_in);

        s->nb_hw_voices_in = 0;

    }



    {

        int def;

        drvname = audio_get_conf_str ("QEMU_AUDIO_DRV", NULL, &def);

    }



    if (drvname) {

        int found = 0;



        for (i = 0; i < ARRAY_SIZE (drvtab); i++) {

            if (!strcmp (drvname, drvtab[i]->name)) {

                done = !audio_driver_init (s, drvtab[i]);

                found = 1;

                break;

            }

        }



        if (!found) {

            dolog ("Unknown audio driver `%s'\n", drvname);

            dolog ("Run with -audio-help to list available drivers\n");

        }

    }



    if (!done) {

        for (i = 0; !done && i < ARRAY_SIZE (drvtab); i++) {

            if (drvtab[i]->can_be_default) {

                done = !audio_driver_init (s, drvtab[i]);

            }

        }

    }



    if (!done) {

        done = !audio_driver_init (s, &no_audio_driver);

        if (!done) {

            hw_error("Could not initialize audio subsystem\n");

        }

        else {

            dolog ("warning: Using timer based audio emulation\n");

        }

    }



    if (conf.period.hertz <= 0) {

        if (conf.period.hertz < 0) {

            dolog ("warning: Timer period is negative - %d "

                   "treating as zero\n",

                   conf.period.hertz);

        }

        conf.period.ticks = 1;

    } else {

        conf.period.ticks =

            muldiv64 (1, get_ticks_per_sec (), conf.period.hertz);

    }



    e = qemu_add_vm_change_state_handler (audio_vm_change_state_handler, s);

    if (!e) {

        dolog ("warning: Could not register change state handler\n"

               "(Audio can continue looping even after stopping the VM)\n");

    }



    QLIST_INIT (&s->card_head);

    vmstate_register (NULL, 0, &vmstate_audio, s);

}
