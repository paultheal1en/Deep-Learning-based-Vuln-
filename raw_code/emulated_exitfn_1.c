static int emulated_exitfn(CCIDCardState *base)

{

    EmulatedState *card = DO_UPCAST(EmulatedState, base, base);

    VEvent *vevent = vevent_new(VEVENT_LAST, NULL, NULL);



    vevent_queue_vevent(vevent); /* stop vevent thread */

    qemu_mutex_lock(&card->apdu_thread_quit_mutex);

    card->quit_apdu_thread = 1; /* stop handle_apdu thread */

    qemu_cond_signal(&card->handle_apdu_cond);

    qemu_cond_wait(&card->apdu_thread_quit_cond,

                      &card->apdu_thread_quit_mutex);

    /* handle_apdu thread stopped, can destroy all of it's mutexes */

    qemu_cond_destroy(&card->handle_apdu_cond);

    qemu_cond_destroy(&card->apdu_thread_quit_cond);

    qemu_mutex_destroy(&card->apdu_thread_quit_mutex);

    qemu_mutex_destroy(&card->handle_apdu_mutex);

    qemu_mutex_destroy(&card->vreader_mutex);

    qemu_mutex_destroy(&card->event_list_mutex);

    return 0;

}
