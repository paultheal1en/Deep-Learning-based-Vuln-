static int os_host_main_loop_wait(uint32_t timeout)

{

    GMainContext *context = g_main_context_default();

    int select_ret, g_poll_ret, ret, i;

    PollingEntry *pe;

    WaitObjects *w = &wait_objects;

    gint poll_timeout;

    static struct timeval tv0;



    /* XXX: need to suppress polling by better using win32 events */

    ret = 0;

    for (pe = first_polling_entry; pe != NULL; pe = pe->next) {

        ret |= pe->func(pe->opaque);

    }

    if (ret != 0) {

        return ret;

    }



    g_main_context_prepare(context, &max_priority);

    n_poll_fds = g_main_context_query(context, max_priority, &poll_timeout,

                                      poll_fds, ARRAY_SIZE(poll_fds));

    g_assert(n_poll_fds <= ARRAY_SIZE(poll_fds));



    for (i = 0; i < w->num; i++) {

        poll_fds[n_poll_fds + i].fd = (DWORD_PTR)w->events[i];

        poll_fds[n_poll_fds + i].events = G_IO_IN;

    }



    if (poll_timeout < 0 || timeout < poll_timeout) {

        poll_timeout = timeout;

    }



    qemu_mutex_unlock_iothread();

    g_poll_ret = g_poll(poll_fds, n_poll_fds + w->num, poll_timeout);

    qemu_mutex_lock_iothread();

    if (g_poll_ret > 0) {

        for (i = 0; i < w->num; i++) {

            w->revents[i] = poll_fds[n_poll_fds + i].revents;

        }

        for (i = 0; i < w->num; i++) {

            if (w->revents[i] && w->func[i]) {

                w->func[i](w->opaque[i]);

            }

        }

    }



    if (g_main_context_check(context, max_priority, poll_fds, n_poll_fds)) {

        g_main_context_dispatch(context);

    }



    /* Call select after g_poll to avoid a useless iteration and therefore

     * improve socket latency.

     */



    if (nfds >= 0) {

        select_ret = select(nfds + 1, &rfds, &wfds, &xfds, &tv0);

        if (select_ret != 0) {

            timeout = 0;

        }

    }



    return select_ret || g_poll_ret;

}
