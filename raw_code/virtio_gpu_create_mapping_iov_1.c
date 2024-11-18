int virtio_gpu_create_mapping_iov(struct virtio_gpu_resource_attach_backing *ab,

                                  struct virtio_gpu_ctrl_command *cmd,

                                  struct iovec **iov)

{

    struct virtio_gpu_mem_entry *ents;

    size_t esize, s;

    int i;



    if (ab->nr_entries > 16384) {

        qemu_log_mask(LOG_GUEST_ERROR,

                      "%s: nr_entries is too big (%d > 16384)\n",

                      __func__, ab->nr_entries);

        return -1;

    }



    esize = sizeof(*ents) * ab->nr_entries;

    ents = g_malloc(esize);

    s = iov_to_buf(cmd->elem.out_sg, cmd->elem.out_num,

                   sizeof(*ab), ents, esize);

    if (s != esize) {

        qemu_log_mask(LOG_GUEST_ERROR,

                      "%s: command data size incorrect %zu vs %zu\n",

                      __func__, s, esize);

        g_free(ents);

        return -1;

    }



    *iov = g_malloc0(sizeof(struct iovec) * ab->nr_entries);

    for (i = 0; i < ab->nr_entries; i++) {

        hwaddr len = ents[i].length;

        (*iov)[i].iov_len = ents[i].length;

        (*iov)[i].iov_base = cpu_physical_memory_map(ents[i].addr, &len, 1);

        if (!(*iov)[i].iov_base || len != ents[i].length) {

            qemu_log_mask(LOG_GUEST_ERROR, "%s: failed to map MMIO memory for"

                          " resource %d element %d\n",

                          __func__, ab->resource_id, i);

            virtio_gpu_cleanup_mapping_iov(*iov, i);

            g_free(ents);

            g_free(*iov);

            *iov = NULL;

            return -1;

        }

    }

    g_free(ents);

    return 0;

}
