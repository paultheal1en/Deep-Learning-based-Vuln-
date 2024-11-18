static void run_dependent_requests(QCowL2Meta *m)

{

    QCowAIOCB *req;

    QCowAIOCB *next;



    /* Take the request off the list of running requests */

    if (m->nb_clusters != 0) {

        LIST_REMOVE(m, next_in_flight);

    }



    /*

     * Restart all dependent requests.

     * Can't use LIST_FOREACH here - the next link might not be the same

     * any more after the callback  (request could depend on a different

     * request now)

     */

    for (req = m->dependent_requests.lh_first; req != NULL; req = next) {

        next = req->next_depend.le_next;

        qcow_aio_write_cb(req, 0);

    }



    /* Empty the list for the next part of the request */

    LIST_INIT(&m->dependent_requests);

}
