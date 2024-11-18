ssize_t nbd_wr_syncv(QIOChannel *ioc,

                     struct iovec *iov,

                     size_t niov,

                     size_t length,

                     bool do_read,

                     Error **errp)

{

    ssize_t done = 0;

    struct iovec *local_iov = g_new(struct iovec, niov);

    struct iovec *local_iov_head = local_iov;

    unsigned int nlocal_iov = niov;



    nlocal_iov = iov_copy(local_iov, nlocal_iov, iov, niov, 0, length);



    while (nlocal_iov > 0) {

        ssize_t len;

        if (do_read) {

            len = qio_channel_readv(ioc, local_iov, nlocal_iov, errp);

        } else {

            len = qio_channel_writev(ioc, local_iov, nlocal_iov, errp);

        }

        if (len == QIO_CHANNEL_ERR_BLOCK) {

            /* errp should not be set */

            assert(qemu_in_coroutine());

            qio_channel_yield(ioc, do_read ? G_IO_IN : G_IO_OUT);

            continue;

        }

        if (len < 0) {

            done = -EIO;

            goto cleanup;

        }



        if (do_read && len == 0) {

            break;

        }



        iov_discard_front(&local_iov, &nlocal_iov, len);

        done += len;

    }



 cleanup:

    g_free(local_iov_head);

    return done;

}
