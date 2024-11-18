static void qio_channel_command_finalize(Object *obj)

{

    QIOChannelCommand *ioc = QIO_CHANNEL_COMMAND(obj);

    if (ioc->readfd != -1) {

        close(ioc->readfd);

        ioc->readfd = -1;

    }

    if (ioc->writefd != -1) {

        close(ioc->writefd);

        ioc->writefd = -1;

    }

    if (ioc->pid > 0) {

#ifndef WIN32

        qio_channel_command_abort(ioc, NULL);

#endif

    }

}
