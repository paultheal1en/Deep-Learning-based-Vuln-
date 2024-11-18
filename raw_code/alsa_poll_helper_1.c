static int alsa_poll_helper (snd_pcm_t *handle, struct pollhlp *hlp, int mask)

{

    int i, count, err;

    struct pollfd *pfds;



    count = snd_pcm_poll_descriptors_count (handle);

    if (count <= 0) {

        dolog ("Could not initialize poll mode\n"

               "Invalid number of poll descriptors %d\n", count);

        return -1;

    }



    pfds = audio_calloc ("alsa_poll_helper", count, sizeof (*pfds));

    if (!pfds) {

        dolog ("Could not initialize poll mode\n");

        return -1;

    }



    err = snd_pcm_poll_descriptors (handle, pfds, count);

    if (err < 0) {

        alsa_logerr (err, "Could not initialize poll mode\n"

                     "Could not obtain poll descriptors\n");

        g_free (pfds);

        return -1;

    }



    for (i = 0; i < count; ++i) {

        if (pfds[i].events & POLLIN) {

            err = qemu_set_fd_handler (pfds[i].fd, alsa_poll_handler,

                                       NULL, hlp);

        }

        if (pfds[i].events & POLLOUT) {

            if (conf.verbose) {

                dolog ("POLLOUT %d %d\n", i, pfds[i].fd);

            }

            err = qemu_set_fd_handler (pfds[i].fd, NULL,

                                       alsa_poll_handler, hlp);

        }

        if (conf.verbose) {

            dolog ("Set handler events=%#x index=%d fd=%d err=%d\n",

                   pfds[i].events, i, pfds[i].fd, err);

        }



        if (err) {

            dolog ("Failed to set handler events=%#x index=%d fd=%d err=%d\n",

                   pfds[i].events, i, pfds[i].fd, err);



            while (i--) {

                qemu_set_fd_handler (pfds[i].fd, NULL, NULL, NULL);

            }

            g_free (pfds);

            return -1;

        }

    }

    hlp->pfds = pfds;

    hlp->count = count;

    hlp->handle = handle;

    hlp->mask = mask;

    return 0;

}
