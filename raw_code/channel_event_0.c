static void channel_event(int event, SpiceChannelEventInfo *info)

{

    SpiceServerInfo *server = g_malloc0(sizeof(*server));

    SpiceChannel *client = g_malloc0(sizeof(*client));

    server->base = g_malloc0(sizeof(*server->base));

    client->base = g_malloc0(sizeof(*client->base));



    /*

     * Spice server might have called us from spice worker thread

     * context (happens on display channel disconnects).  Spice should

     * not do that.  It isn't that easy to fix it in spice and even

     * when it is fixed we still should cover the already released

     * spice versions.  So detect that we've been called from another

     * thread and grab the iothread lock if so before calling qemu

     * functions.

     */

    bool need_lock = !qemu_thread_is_self(&me);

    if (need_lock) {

        qemu_mutex_lock_iothread();

    }



    if (info->flags & SPICE_CHANNEL_EVENT_FLAG_ADDR_EXT) {

        add_addr_info(client->base, (struct sockaddr *)&info->paddr_ext,

                      info->plen_ext);

        add_addr_info(server->base, (struct sockaddr *)&info->laddr_ext,

                      info->llen_ext);

    } else {

        error_report("spice: %s, extended address is expected",

                     __func__);

    }



    switch (event) {

    case SPICE_CHANNEL_EVENT_CONNECTED:

        qapi_event_send_spice_connected(server->base, client->base, &error_abort);

        break;

    case SPICE_CHANNEL_EVENT_INITIALIZED:

        if (auth) {

            server->has_auth = true;

            server->auth = g_strdup(auth);

        }

        add_channel_info(client, info);

        channel_list_add(info);

        qapi_event_send_spice_initialized(server, client, &error_abort);

        break;

    case SPICE_CHANNEL_EVENT_DISCONNECTED:

        channel_list_del(info);

        qapi_event_send_spice_disconnected(server->base, client->base, &error_abort);

        break;

    default:

        break;

    }



    if (need_lock) {

        qemu_mutex_unlock_iothread();

    }



    qapi_free_SpiceServerInfo(server);

    qapi_free_SpiceChannel(client);

}
