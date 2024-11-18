qcrypto_tls_creds_check_cert_key_purpose(QCryptoTLSCredsX509 *creds,

                                         gnutls_x509_crt_t cert,

                                         const char *certFile,

                                         bool isServer,

                                         Error **errp)

{

    int status;

    size_t i;

    unsigned int purposeCritical;

    unsigned int critical;

    char *buffer = NULL;

    size_t size;

    bool allowClient = false, allowServer = false;



    critical = 0;

    for (i = 0; ; i++) {

        size = 0;

        status = gnutls_x509_crt_get_key_purpose_oid(cert, i, buffer,

                                                     &size, NULL);



        if (status == GNUTLS_E_REQUESTED_DATA_NOT_AVAILABLE) {



            /* If there is no data at all, then we must allow

               client/server to pass */

            if (i == 0) {

                allowServer = allowClient = true;

            }

            break;

        }

        if (status != GNUTLS_E_SHORT_MEMORY_BUFFER) {

            error_setg(errp,

                       "Unable to query certificate %s key purpose: %s",

                       certFile, gnutls_strerror(status));

            return -1;

        }



        buffer = g_new0(char, size);



        status = gnutls_x509_crt_get_key_purpose_oid(cert, i, buffer,

                                                     &size, &purposeCritical);



        if (status < 0) {

            trace_qcrypto_tls_creds_x509_check_key_purpose(

                creds, certFile, status, "<none>", purposeCritical);

            g_free(buffer);

            error_setg(errp,

                       "Unable to query certificate %s key purpose: %s",

                       certFile, gnutls_strerror(status));

            return -1;

        }

        trace_qcrypto_tls_creds_x509_check_key_purpose(

            creds, certFile, status, buffer, purposeCritical);

        if (purposeCritical) {

            critical = true;

        }



        if (g_str_equal(buffer, GNUTLS_KP_TLS_WWW_SERVER)) {

            allowServer = true;

        } else if (g_str_equal(buffer, GNUTLS_KP_TLS_WWW_CLIENT)) {

            allowClient = true;

        } else if (g_str_equal(buffer, GNUTLS_KP_ANY)) {

            allowServer = allowClient = true;

        }



        g_free(buffer);


    }



    if (isServer) {

        if (!allowServer) {

            if (critical) {

                error_setg(errp,

                           "Certificate %s purpose does not allow "

                           "use with a TLS server", certFile);

                return -1;

            }

        }

    } else {

        if (!allowClient) {

            if (critical) {

                error_setg(errp,

                           "Certificate %s purpose does not allow use "

                           "with a TLS client", certFile);

                return -1;

            }

        }

    }



    return 0;

}