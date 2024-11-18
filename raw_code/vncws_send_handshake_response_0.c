static void vncws_send_handshake_response(VncState *vs, const char* key)

{

    char combined_key[WS_CLIENT_KEY_LEN + WS_GUID_LEN + 1];

    unsigned char hash[SHA1_DIGEST_LEN];

    size_t hash_size = sizeof(hash);

    char *accept = NULL, *response = NULL;

    gnutls_datum_t in;

    int ret;



    g_strlcpy(combined_key, key, WS_CLIENT_KEY_LEN + 1);

    g_strlcat(combined_key, WS_GUID, WS_CLIENT_KEY_LEN + WS_GUID_LEN + 1);



    /* hash and encode it */

    in.data = (void *)combined_key;

    in.size = WS_CLIENT_KEY_LEN + WS_GUID_LEN;

    ret = gnutls_fingerprint(GNUTLS_DIG_SHA1, &in, hash, &hash_size);

    if (ret == GNUTLS_E_SUCCESS && hash_size <= SHA1_DIGEST_LEN) {

        accept = g_base64_encode(hash, hash_size);

    }

    if (accept == NULL) {

        VNC_DEBUG("Hashing Websocket combined key failed\n");

        vnc_client_error(vs);

        return;

    }



    response = g_strdup_printf(WS_HANDSHAKE, accept);

    vnc_client_write_buf(vs, (const uint8_t *)response, strlen(response));



    g_free(accept);

    g_free(response);



    vs->encode_ws = 1;

    vnc_init_state(vs);

}
