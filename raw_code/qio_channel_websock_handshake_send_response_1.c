static int qio_channel_websock_handshake_send_response(QIOChannelWebsock *ioc,

                                                       const char *key,

                                                       Error **errp)

{

    char combined_key[QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN +

                      QIO_CHANNEL_WEBSOCK_GUID_LEN + 1];

    char *accept = NULL, *response = NULL;

    size_t responselen;



    g_strlcpy(combined_key, key, QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN + 1);

    g_strlcat(combined_key, QIO_CHANNEL_WEBSOCK_GUID,

              QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN +

              QIO_CHANNEL_WEBSOCK_GUID_LEN + 1);



    /* hash and encode it */

    if (qcrypto_hash_base64(QCRYPTO_HASH_ALG_SHA1,

                            combined_key,

                            QIO_CHANNEL_WEBSOCK_CLIENT_KEY_LEN +

                            QIO_CHANNEL_WEBSOCK_GUID_LEN,

                            &accept,

                            errp) < 0) {

        return -1;

    }



    response = g_strdup_printf(QIO_CHANNEL_WEBSOCK_HANDSHAKE_RESPONSE, accept);

    responselen = strlen(response);

    buffer_reserve(&ioc->encoutput, responselen);

    buffer_append(&ioc->encoutput, response, responselen);



    g_free(accept);

    g_free(response);



    return 0;

}
