static int protocol_client_vencrypt_auth(VncState *vs, uint8_t *data, size_t len)

{

    int auth = read_u32(data, 0);



    if (auth != vs->vd->subauth) {

        VNC_DEBUG("Rejecting auth %d\n", auth);

        vnc_write_u8(vs, 0); /* Reject auth */

        vnc_flush(vs);

        vnc_client_error(vs);

    } else {

        VNC_DEBUG("Accepting auth %d, setting up TLS for handshake\n", auth);

        vnc_write_u8(vs, 1); /* Accept auth */

        vnc_flush(vs);



        if (vnc_tls_client_setup(vs, NEED_X509_AUTH(vs)) < 0) {

            VNC_DEBUG("Failed to setup TLS\n");

            return 0;

        }



        VNC_DEBUG("Start TLS VeNCrypt handshake process\n");

        if (vnc_start_vencrypt_handshake(vs) < 0) {

            VNC_DEBUG("Failed to start TLS handshake\n");

            return 0;

        }

    }

    return 0;

}
