static guint io_add_watch_poll(GIOChannel *channel,

                               IOCanReadHandler *fd_can_read,

                               GIOFunc fd_read,

                               gpointer user_data)

{

    IOWatchPoll *iwp;



    iwp = (IOWatchPoll *) g_source_new(&io_watch_poll_funcs, sizeof(IOWatchPoll));

    iwp->fd_can_read = fd_can_read;

    iwp->opaque = user_data;

    iwp->src = g_io_create_watch(channel, G_IO_IN | G_IO_ERR | G_IO_HUP);

    g_source_set_callback(iwp->src, (GSourceFunc)fd_read, user_data, NULL);



    return g_source_attach(&iwp->parent, NULL);

}
