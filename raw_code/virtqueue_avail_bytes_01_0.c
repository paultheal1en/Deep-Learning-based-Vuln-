int virtqueue_avail_bytes(VirtQueue *vq, unsigned int in_bytes,

                          unsigned int out_bytes)

{

    unsigned int in_total, out_total;



    virtqueue_get_avail_bytes(vq, &in_total, &out_total);

    if ((in_bytes && in_bytes < in_total)

        || (out_bytes && out_bytes < out_total)) {

        return 1;

    }

    return 0;

}
