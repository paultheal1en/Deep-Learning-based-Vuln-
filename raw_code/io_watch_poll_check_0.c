static gboolean io_watch_poll_check(GSource *source)

{

    IOWatchPoll *iwp = io_watch_poll_from_source(source);



    if (iwp->max_size == 0) {

        return FALSE;

    }



    return g_io_watch_funcs.check(source);

}
