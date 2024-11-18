static void virtio_balloon_handle_output(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIOBalloon *s = VIRTIO_BALLOON(vdev);

    VirtQueueElement elem;

    MemoryRegionSection section;



    while (virtqueue_pop(vq, &elem)) {

        size_t offset = 0;

        uint32_t pfn;



        while (iov_to_buf(elem.out_sg, elem.out_num, offset, &pfn, 4) == 4) {

            ram_addr_t pa;

            ram_addr_t addr;

            int p = virtio_ldl_p(vdev, &pfn);



            pa = (ram_addr_t) p << VIRTIO_BALLOON_PFN_SHIFT;

            offset += 4;



            /* FIXME: remove get_system_memory(), but how? */

            section = memory_region_find(get_system_memory(), pa, 1);

            if (!int128_nz(section.size) || !memory_region_is_ram(section.mr))

                continue;



            trace_virtio_balloon_handle_output(memory_region_name(section.mr),

                                               pa);

            /* Using memory_region_get_ram_ptr is bending the rules a bit, but

               should be OK because we only want a single page.  */

            addr = section.offset_within_region;

            balloon_page(memory_region_get_ram_ptr(section.mr) + addr,

                         !!(vq == s->dvq));

            memory_region_unref(section.mr);

        }



        virtqueue_push(vq, &elem, offset);

        virtio_notify(vdev, vq);

    }

}
