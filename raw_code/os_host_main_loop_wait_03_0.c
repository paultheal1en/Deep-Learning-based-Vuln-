static int os_host_main_loop_wait(int timeout)

{

    int ret, i;

    PollingEntry *pe;

    WaitObjects *w = &wait_objects;

    static struct timeval tv0;



    /* XXX: need to suppress polling by better using win32 events */

    ret = 0;

    for (pe = first_polling_entry; pe != NULL; pe = pe->next) {

        ret |= pe->func(pe->opaque);

    }

    if (ret != 0) {

        return ret;

    }



    if (nfds >= 0) {

        ret = select(nfds + 1, &rfds, &wfds, &xfds, &tv0);

        if (ret != 0) {

            timeout = 0;

        }

    }



    for (i = 0; i < w->num; i++) {

        poll_fds[i].fd = (DWORD) w->events[i];

        poll_fds[i].events = G_IO_IN;

    }



    qemu_mutex_unlock_iothread();

    ret = g_poll(poll_fds, w->num, timeout);

    qemu_mutex_lock_iothread();

    if (ret > 0) {

        for (i = 0; i < w->num; i++) {

            w->revents[i] = poll_fds[i].revents;

        }

        for (i = 0; i < w->num; i++) {

            if (w->revents[i] && w->func[i]) {

                w->func[i](w->opaque[i]);

            }

        }

    }



    /* If an edge-triggered socket event occurred, select will return a

     * positive result on the next iteration.  We do not need to do anything

     * here.

     */



    return ret;

}
