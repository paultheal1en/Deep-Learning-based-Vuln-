static ssize_t vnc_client_write_tls(gnutls_session_t *session,

                                    const uint8_t *data,

                                    size_t datalen)

{

    ssize_t ret = gnutls_write(*session, data, datalen);

    if (ret < 0) {

        if (ret == GNUTLS_E_AGAIN) {

            errno = EAGAIN;

        } else {

            errno = EIO;

        }

        ret = -1;

    }

    return ret;

}
