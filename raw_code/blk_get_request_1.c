static int blk_get_request(struct XenBlkDev *blkdev, struct ioreq *ioreq, RING_IDX rc)

{

    switch (blkdev->protocol) {

    case BLKIF_PROTOCOL_NATIVE:

        memcpy(&ioreq->req, RING_GET_REQUEST(&blkdev->rings.native, rc),

               sizeof(ioreq->req));

        break;

    case BLKIF_PROTOCOL_X86_32:

        blkif_get_x86_32_req(&ioreq->req,

                             RING_GET_REQUEST(&blkdev->rings.x86_32_part, rc));

        break;

    case BLKIF_PROTOCOL_X86_64:

        blkif_get_x86_64_req(&ioreq->req,

                             RING_GET_REQUEST(&blkdev->rings.x86_64_part, rc));

        break;

    }



    return 0;

}