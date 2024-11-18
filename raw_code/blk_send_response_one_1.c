static int blk_send_response_one(struct ioreq *ioreq)

{

    struct XenBlkDev  *blkdev = ioreq->blkdev;

    int               send_notify   = 0;

    int               have_requests = 0;

    blkif_response_t  resp;

    void              *dst;



    resp.id        = ioreq->req.id;

    resp.operation = ioreq->req.operation;

    resp.status    = ioreq->status;



    /* Place on the response ring for the relevant domain. */

    switch (blkdev->protocol) {

    case BLKIF_PROTOCOL_NATIVE:

        dst = RING_GET_RESPONSE(&blkdev->rings.native, blkdev->rings.native.rsp_prod_pvt);

        break;

    case BLKIF_PROTOCOL_X86_32:

        dst = RING_GET_RESPONSE(&blkdev->rings.x86_32_part,

                                blkdev->rings.x86_32_part.rsp_prod_pvt);

        break;

    case BLKIF_PROTOCOL_X86_64:

        dst = RING_GET_RESPONSE(&blkdev->rings.x86_64_part,

                                blkdev->rings.x86_64_part.rsp_prod_pvt);

        break;

    default:

        dst = NULL;


    }

    memcpy(dst, &resp, sizeof(resp));

    blkdev->rings.common.rsp_prod_pvt++;



    RING_PUSH_RESPONSES_AND_CHECK_NOTIFY(&blkdev->rings.common, send_notify);

    if (blkdev->rings.common.rsp_prod_pvt == blkdev->rings.common.req_cons) {

        /*

         * Tail check for pending requests. Allows frontend to avoid

         * notifications if requests are already in flight (lower

         * overheads and promotes batching).

         */

        RING_FINAL_CHECK_FOR_REQUESTS(&blkdev->rings.common, have_requests);

    } else if (RING_HAS_UNCONSUMED_REQUESTS(&blkdev->rings.common)) {

        have_requests = 1;

    }



    if (have_requests) {

        blkdev->more_work++;

    }

    return send_notify;

}