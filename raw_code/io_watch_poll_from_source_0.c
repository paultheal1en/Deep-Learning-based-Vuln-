static IOWatchPoll *io_watch_poll_from_source(GSource *source)

{

    IOWatchPoll *i;



    QTAILQ_FOREACH(i, &io_watch_poll_list, node) {

        if (i->src == source) {

            return i;

        }

    }



    return NULL;

}
