void virtqueue_get_avail_bytes(VirtQueue *vq, unsigned int *in_bytes,

                               unsigned int *out_bytes,

                               unsigned max_in_bytes, unsigned max_out_bytes)

{

    VirtIODevice *vdev = vq->vdev;

    unsigned int max, idx;

    unsigned int total_bufs, in_total, out_total;

    VRingMemoryRegionCaches *caches;

    MemoryRegionCache indirect_desc_cache = MEMORY_REGION_CACHE_INVALID;

    int64_t len = 0;

    int rc;



    if (unlikely(!vq->vring.desc)) {

        if (in_bytes) {

            *in_bytes = 0;

        }

        if (out_bytes) {

            *out_bytes = 0;

        }

        return;

    }



    rcu_read_lock();

    idx = vq->last_avail_idx;

    total_bufs = in_total = out_total = 0;



    max = vq->vring.num;

    caches = atomic_rcu_read(&vq->vring.caches);

    if (caches->desc.len < max * sizeof(VRingDesc)) {

        virtio_error(vdev, "Cannot map descriptor ring");

        goto err;

    }



    while ((rc = virtqueue_num_heads(vq, idx)) > 0) {

        MemoryRegionCache *desc_cache = &caches->desc;

        unsigned int num_bufs;

        VRingDesc desc;

        unsigned int i;



        num_bufs = total_bufs;



        if (!virtqueue_get_head(vq, idx++, &i)) {

            goto err;

        }



        vring_desc_read(vdev, &desc, desc_cache, i);



        if (desc.flags & VRING_DESC_F_INDIRECT) {

            if (desc.len % sizeof(VRingDesc)) {

                virtio_error(vdev, "Invalid size for indirect buffer table");

                goto err;

            }



            /* If we've got too many, that implies a descriptor loop. */

            if (num_bufs >= max) {

                virtio_error(vdev, "Looped descriptor");

                goto err;

            }



            /* loop over the indirect descriptor table */

            len = address_space_cache_init(&indirect_desc_cache,

                                           vdev->dma_as,

                                           desc.addr, desc.len, false);

            desc_cache = &indirect_desc_cache;

            if (len < desc.len) {

                virtio_error(vdev, "Cannot map indirect buffer");

                goto err;

            }



            max = desc.len / sizeof(VRingDesc);

            num_bufs = i = 0;

            vring_desc_read(vdev, &desc, desc_cache, i);

        }



        do {

            /* If we've got too many, that implies a descriptor loop. */

            if (++num_bufs > max) {

                virtio_error(vdev, "Looped descriptor");

                goto err;

            }



            if (desc.flags & VRING_DESC_F_WRITE) {

                in_total += desc.len;

            } else {

                out_total += desc.len;

            }

            if (in_total >= max_in_bytes && out_total >= max_out_bytes) {

                goto done;

            }



            rc = virtqueue_read_next_desc(vdev, &desc, desc_cache, max, &i);

        } while (rc == VIRTQUEUE_READ_DESC_MORE);



        if (rc == VIRTQUEUE_READ_DESC_ERROR) {

            goto err;

        }



        if (desc_cache == &indirect_desc_cache) {

            address_space_cache_destroy(&indirect_desc_cache);

            total_bufs++;

        } else {

            total_bufs = num_bufs;

        }

    }



    if (rc < 0) {

        goto err;

    }



done:

    address_space_cache_destroy(&indirect_desc_cache);

    if (in_bytes) {

        *in_bytes = in_total;

    }

    if (out_bytes) {

        *out_bytes = out_total;

    }

    rcu_read_unlock();

    return;



err:

    in_total = out_total = 0;

    goto done;

}
