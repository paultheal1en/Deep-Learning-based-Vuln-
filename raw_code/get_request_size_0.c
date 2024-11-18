static size_t get_request_size(VirtQueue *vq)

{

    unsigned int in, out;



    virtqueue_get_avail_bytes(vq, &in, &out);

    return in;

}
