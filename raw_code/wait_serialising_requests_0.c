static bool coroutine_fn wait_serialising_requests(BdrvTrackedRequest *self)

{

    BlockDriverState *bs = self->bs;

    BdrvTrackedRequest *req;

    bool retry;

    bool waited = false;



    if (!bs->serialising_in_flight) {

        return false;

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



                /* If the request is already (indirectly) waiting for us, or

                 * will wait for us as soon as it wakes up, then just go on

                 * (instead of producing a deadlock in the former case). */

                if (!req->waiting_for) {

                    self->waiting_for = req;

                    qemu_co_queue_wait(&req->wait_queue);

                    self->waiting_for = NULL;

                    retry = true;

                    waited = true;

                    break;

                }

            }

        }

    } while (retry);



    return waited;

}
