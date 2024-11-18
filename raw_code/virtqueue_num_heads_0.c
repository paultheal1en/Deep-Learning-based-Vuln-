static int virtqueue_num_heads(VirtQueue *vq, unsigned int idx)

{

    uint16_t num_heads = vring_avail_idx(vq) - idx;



    /* Check it isn't doing very strange things with descriptor numbers. */

    if (num_heads > vq->vring.num) {

        error_report("Guest moved used index from %u to %u",

                     idx, vring_avail_idx(vq));

        exit(1);

    }

    /* On success, callers read a descriptor at vq->last_avail_idx.

     * Make sure descriptor read does not bypass avail index read. */

    if (num_heads) {

        smp_rmb();

    }



    return num_heads;

}
