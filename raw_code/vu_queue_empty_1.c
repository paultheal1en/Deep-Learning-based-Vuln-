vu_queue_empty(VuDev *dev, VuVirtq *vq)

{

    if (vq->shadow_avail_idx != vq->last_avail_idx) {

        return 0;

    }



    return vring_avail_idx(vq) == vq->last_avail_idx;

}
