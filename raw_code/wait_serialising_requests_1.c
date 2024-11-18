static void coroutine_fn wait_serialising_requests(BdrvTrackedRequest *self)

{

    BlockDriverState *bs = self->bs;

    BdrvTrackedRequest *req;

    bool retry;



    if (!bs->serialising_in_flight) {

        return;

    }



    do {

        retry = false;

        QLIST_FOREACH(req, &bs->tracked_requests, list) {

            if (req == self || (!req->serialising && !self->serialising)) {

                continue;

            }

            if (tracked_request_overlaps(req, self->overlap_offset,

                                         self->overlap_bytes))

            {

                /* Hitting this means there was a reentrant request, for

                 * example, a block driver issuing nested requests.  This must

                 * never happen since it means deadlock.

                 */

                assert(qemu_coroutine_self() != req->co);



                qemu_co_queue_wait(&req->wait_queue);

                retry = true;

                break;

            }

        }

    } while (retry);

}
