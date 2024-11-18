qcrypto_tls_session_check_credentials(QCryptoTLSSession *session,

                                      Error **errp)

{

    if (object_dynamic_cast(OBJECT(session->creds),

                            TYPE_QCRYPTO_TLS_CREDS_ANON)) {

        return 0;

    } else if (object_dynamic_cast(OBJECT(session->creds),

                            TYPE_QCRYPTO_TLS_CREDS_X509)) {

        if (session->creds->verifyPeer) {

            return qcrypto_tls_session_check_certificate(session,

                                                         errp);

        } else {

            return 0;

        }

    } else {

        error_setg(errp, "Unexpected credential type %s",

                   object_get_typename(OBJECT(session->creds)));

        return -1;

    }

}
