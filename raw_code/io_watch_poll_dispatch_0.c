static gboolean io_watch_poll_dispatch(GSource *source, GSourceFunc callback,

                                       gpointer user_data)

{

    return g_io_watch_funcs.dispatch(source, callback, user_data);

}
