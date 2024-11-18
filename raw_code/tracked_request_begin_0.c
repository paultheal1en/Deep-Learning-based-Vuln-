static void tracked_request_begin(BdrvTrackedRequest *req,

                                  BlockDriverState *bs,

                                  int64_t offset,

                                  unsigned int bytes, bool is_write)

{

    *req = (BdrvTrackedRequest){

        .bs = bs,

        .offset         = offset,

        .bytes          = bytes,

        .is_write       = is_write,

        .co             = qemu_coroutine_self(),

        .serialising    = false,

        .overlap_offset = offset,

        .overlap_bytes  = bytes,

    };



    qemu_co_queue_init(&req->wait_queue);



    QLIST_INSERT_HEAD(&bs->tracked_requests, req, list);

}
