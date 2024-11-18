static void virtqueue_unmap_sg(VirtQueue *vq, const VirtQueueElement *elem,

                               unsigned int len)

{

    unsigned int offset;

    int i;



    offset = 0;

    for (i = 0; i < elem->in_num; i++) {

        size_t size = MIN(len - offset, elem->in_sg[i].iov_len);



        cpu_physical_memory_unmap(elem->in_sg[i].iov_base,

                                  elem->in_sg[i].iov_len,

                                  1, size);



        offset += size;

    }



    for (i = 0; i < elem->out_num; i++)

        cpu_physical_memory_unmap(elem->out_sg[i].iov_base,

                                  elem->out_sg[i].iov_len,

                                  0, elem->out_sg[i].iov_len);

}
