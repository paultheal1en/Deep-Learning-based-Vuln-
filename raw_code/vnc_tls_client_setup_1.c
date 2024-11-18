int vnc_tls_client_setup(VncState *vs,

                         int needX509Creds) {

    VNC_DEBUG("Do TLS setup\n");

    if (vnc_tls_initialize() < 0) {

        VNC_DEBUG("Failed to init TLS\n");

        vnc_client_error(vs);

        return -1;

    }

    if (vs->tls.session == NULL) {

        if (gnutls_init(&vs->tls.session, GNUTLS_SERVER) < 0) {

            vnc_client_error(vs);

            return -1;

        }



        if (gnutls_set_default_priority(vs->tls.session) < 0) {

            gnutls_deinit(vs->tls.session);

            vs->tls.session = NULL;

            vnc_client_error(vs);

            return -1;

        }



        if (vnc_set_gnutls_priority(vs->tls.session, needX509Creds) < 0) {

            gnutls_deinit(vs->tls.session);

            vs->tls.session = NULL;

            vnc_client_error(vs);

            return -1;

        }



        if (needX509Creds) {

            gnutls_certificate_server_credentials x509_cred =

                vnc_tls_initialize_x509_cred(vs->vd);

            if (!x509_cred) {

                gnutls_deinit(vs->tls.session);

                vs->tls.session = NULL;

                vnc_client_error(vs);

                return -1;

            }

            if (gnutls_credentials_set(vs->tls.session,

                                       GNUTLS_CRD_CERTIFICATE, x509_cred) < 0) {

                gnutls_deinit(vs->tls.session);

                vs->tls.session = NULL;

                gnutls_certificate_free_credentials(x509_cred);

                vnc_client_error(vs);

                return -1;

            }

            if (vs->vd->tls.x509verify) {

                VNC_DEBUG("Requesting a client certificate\n");

                gnutls_certificate_server_set_request(vs->tls.session,

                                                      GNUTLS_CERT_REQUEST);

            }



        } else {

            gnutls_anon_server_credentials_t anon_cred =

                vnc_tls_initialize_anon_cred();

            if (!anon_cred) {

                gnutls_deinit(vs->tls.session);

                vs->tls.session = NULL;

                vnc_client_error(vs);

                return -1;

            }

            if (gnutls_credentials_set(vs->tls.session,

                                       GNUTLS_CRD_ANON, anon_cred) < 0) {

                gnutls_deinit(vs->tls.session);

                vs->tls.session = NULL;

                gnutls_anon_free_server_credentials(anon_cred);

                vnc_client_error(vs);

                return -1;

            }

        }



        gnutls_transport_set_ptr(vs->tls.session, (gnutls_transport_ptr_t)vs);

        gnutls_transport_set_push_function(vs->tls.session, vnc_tls_push);

        gnutls_transport_set_pull_function(vs->tls.session, vnc_tls_pull);

    }

    return 0;

}
