static void tracked_request_end(BdrvTrackedRequest *req)

{

    if (req->serialising) {

        req->bs->serialising_in_flight--;

    }



    QLIST_REMOVE(req, list);

    qemu_co_queue_restart_all(&req->wait_queue);

}
