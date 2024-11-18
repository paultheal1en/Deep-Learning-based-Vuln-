vu_queue_flush(VuDev *dev, VuVirtq *vq, unsigned int count)

{

    uint16_t old, new;



    if (unlikely(dev->broken)) {

        return;

    }



    /* Make sure buffer is written before we update index. */

    smp_wmb();



    old = vq->used_idx;

    new = old + count;

    vring_used_idx_set(dev, vq, new);

    vq->inuse -= count;

    if (unlikely((int16_t)(new - vq->signalled_used) < (uint16_t)(new - old))) {

        vq->signalled_used_valid = false;

    }

}
