static gboolean ga_channel_open(GAChannel *c, const gchar *path, GAChannelMethod method)

{

    int ret;

    c->method = method;



    switch (c->method) {

    case GA_CHANNEL_VIRTIO_SERIAL: {

        int fd = qemu_open(path, O_RDWR | O_NONBLOCK

#ifndef CONFIG_SOLARIS

                           | O_ASYNC

#endif

                           );

        if (fd == -1) {

            g_critical("error opening channel: %s", strerror(errno));

            exit(EXIT_FAILURE);

        }

#ifdef CONFIG_SOLARIS

        ret = ioctl(fd, I_SETSIG, S_OUTPUT | S_INPUT | S_HIPRI);

        if (ret == -1) {

            g_critical("error setting event mask for channel: %s",

                       strerror(errno));

            exit(EXIT_FAILURE);

        }

#endif

        ret = ga_channel_client_add(c, fd);

        if (ret) {

            g_critical("error adding channel to main loop");


            return false;

        }

        break;

    }

    case GA_CHANNEL_ISA_SERIAL: {

        struct termios tio;

        int fd = qemu_open(path, O_RDWR | O_NOCTTY | O_NONBLOCK);

        if (fd == -1) {

            g_critical("error opening channel: %s", strerror(errno));

            exit(EXIT_FAILURE);

        }

        tcgetattr(fd, &tio);

        /* set up serial port for non-canonical, dumb byte streaming */

        tio.c_iflag &= ~(IGNBRK | BRKINT | IGNPAR | PARMRK | INPCK | ISTRIP |

                         INLCR | IGNCR | ICRNL | IXON | IXOFF | IXANY |

                         IMAXBEL);

        tio.c_oflag = 0;

        tio.c_lflag = 0;

        tio.c_cflag |= GA_CHANNEL_BAUDRATE_DEFAULT;

        /* 1 available byte min or reads will block (we'll set non-blocking

         * elsewhere, else we have to deal with read()=0 instead)

         */

        tio.c_cc[VMIN] = 1;

        tio.c_cc[VTIME] = 0;

        /* flush everything waiting for read/xmit, it's garbage at this point */

        tcflush(fd, TCIFLUSH);

        tcsetattr(fd, TCSANOW, &tio);

        ret = ga_channel_client_add(c, fd);

        if (ret) {

            g_error("error adding channel to main loop");

        }

        break;

    }

    case GA_CHANNEL_UNIX_LISTEN: {

        Error *local_err = NULL;

        int fd = unix_listen(path, NULL, strlen(path), &local_err);

        if (local_err != NULL) {

            g_critical("%s", error_get_pretty(local_err));

            error_free(local_err);

            return false;

        }

        ga_channel_listen_add(c, fd, true);

        break;

    }

    default:

        g_critical("error binding/listening to specified socket");

        return false;

    }



    return true;

}