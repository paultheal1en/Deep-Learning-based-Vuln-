static void ga_channel_client_close(GAChannel *c)

{

    g_assert(c->client_channel);

    g_io_channel_shutdown(c->client_channel, true, NULL);

    g_io_channel_unref(c->client_channel);

    c->client_channel = NULL;

    if (c->method == GA_CHANNEL_UNIX_LISTEN && c->listen_channel) {

        ga_channel_listen_add(c, 0, false);

    }

}
