static ssize_t vnc_tls_pull(gnutls_transport_ptr_t transport,

                            void *data,

                            size_t len) {

    VncState *vs = (VncState *)transport;

    int ret;



 retry:

    ret = qemu_recv(vs->csock, data, len, 0);

    if (ret < 0) {

        if (errno == EINTR)

            goto retry;

        return -1;

    }

    return ret;

}
