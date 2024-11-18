static int poll_rest(gboolean poll_msgs, HANDLE *handles, gint nhandles,

                     GPollFD *fds, guint nfds, gint timeout)

{

    DWORD ready;

    GPollFD *f;

    int recursed_result;



    if (poll_msgs) {

        /* Wait for either messages or handles

         * -> Use MsgWaitForMultipleObjectsEx

         */

        ready = MsgWaitForMultipleObjectsEx(nhandles, handles, timeout,

                                            QS_ALLINPUT, MWMO_ALERTABLE);



        if (ready == WAIT_FAILED) {

            gchar *emsg = g_win32_error_message(GetLastError());

            g_warning("MsgWaitForMultipleObjectsEx failed: %s", emsg);

            g_free(emsg);

        }

    } else if (nhandles == 0) {

        /* No handles to wait for, just the timeout */

        if (timeout == INFINITE) {

            ready = WAIT_FAILED;

        } else {

            SleepEx(timeout, TRUE);

            ready = WAIT_TIMEOUT;

        }

    } else {

        /* Wait for just handles

         * -> Use WaitForMultipleObjectsEx

         */

        ready =

            WaitForMultipleObjectsEx(nhandles, handles, FALSE, timeout, TRUE);

        if (ready == WAIT_FAILED) {

            gchar *emsg = g_win32_error_message(GetLastError());

            g_warning("WaitForMultipleObjectsEx failed: %s", emsg);

            g_free(emsg);

        }

    }



    if (ready == WAIT_FAILED) {

        return -1;

    } else if (ready == WAIT_TIMEOUT || ready == WAIT_IO_COMPLETION) {

        return 0;

    } else if (poll_msgs && ready == WAIT_OBJECT_0 + nhandles) {

        for (f = fds; f < &fds[nfds]; ++f) {

            if (f->fd == G_WIN32_MSG_HANDLE && f->events & G_IO_IN) {

                f->revents |= G_IO_IN;

            }

        }



        /* If we have a timeout, or no handles to poll, be satisfied

         * with just noticing we have messages waiting.

         */

        if (timeout != 0 || nhandles == 0) {

            return 1;

        }



        /* If no timeout and handles to poll, recurse to poll them,

         * too.

         */

        recursed_result = poll_rest(FALSE, handles, nhandles, fds, nfds, 0);

        return (recursed_result == -1) ? -1 : 1 + recursed_result;

    } else if (/* QEMU: removed the following unneeded statement which causes

                * a compiler warning: ready >= WAIT_OBJECT_0 && */

               ready < WAIT_OBJECT_0 + nhandles) {

        for (f = fds; f < &fds[nfds]; ++f) {

            if ((HANDLE) f->fd == handles[ready - WAIT_OBJECT_0]) {

                f->revents = f->events;

            }

        }



        /* If no timeout and polling several handles, recurse to poll

         * the rest of them.

         */

        if (timeout == 0 && nhandles > 1) {

            /* Remove the handle that fired */

            int i;

            if (ready < nhandles - 1) {

                for (i = ready - WAIT_OBJECT_0 + 1; i < nhandles; i++) {

                    handles[i-1] = handles[i];

                }

            }

            nhandles--;

            recursed_result = poll_rest(FALSE, handles, nhandles, fds, nfds, 0);

            return (recursed_result == -1) ? -1 : 1 + recursed_result;

        }

        return 1;

    }



    return 0;

}
