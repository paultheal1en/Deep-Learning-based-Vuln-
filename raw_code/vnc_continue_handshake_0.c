static int vnc_continue_handshake(struct VncState *vs) {

    int ret;



    if ((ret = gnutls_handshake(vs->tls_session)) < 0) {

       if (!gnutls_error_is_fatal(ret)) {

           VNC_DEBUG("Handshake interrupted (blocking)\n");

           if (!gnutls_record_get_direction(vs->tls_session))

               qemu_set_fd_handler(vs->csock, vnc_handshake_io, NULL, vs);

           else

               qemu_set_fd_handler(vs->csock, NULL, vnc_handshake_io, vs);

           return 0;

       }

       VNC_DEBUG("Handshake failed %s\n", gnutls_strerror(ret));

       vnc_client_error(vs);

       return -1;

    }



    if (vs->vd->x509verify) {

	if (vnc_validate_certificate(vs) < 0) {

	    VNC_DEBUG("Client verification failed\n");

	    vnc_client_error(vs);

	    return -1;

	} else {

	    VNC_DEBUG("Client verification passed\n");

	}

    }



    VNC_DEBUG("Handshake done, switching to TLS data mode\n");

    vs->wiremode = VNC_WIREMODE_TLS;

    qemu_set_fd_handler2(vs->csock, NULL, vnc_client_read, vnc_client_write, vs);



    return start_auth_vencrypt_subauth(vs);

}
