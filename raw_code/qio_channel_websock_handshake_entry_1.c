static char *qio_channel_websock_handshake_entry(const char *handshake,

                                                 size_t handshake_len,

                                                 const char *name)

{

    char *begin, *end, *ret = NULL;

    char *line = g_strdup_printf("%s%s: ",

                                 QIO_CHANNEL_WEBSOCK_HANDSHAKE_DELIM,

                                 name);

    begin = g_strstr_len(handshake, handshake_len, line);

    if (begin != NULL) {

        begin += strlen(line);

        end = g_strstr_len(begin, handshake_len - (begin - handshake),

                QIO_CHANNEL_WEBSOCK_HANDSHAKE_DELIM);

        if (end != NULL) {

            ret = g_strndup(begin, end - begin);

        }

    }

    g_free(line);

    return ret;

}
