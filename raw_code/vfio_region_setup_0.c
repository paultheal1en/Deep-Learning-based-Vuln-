int vfio_region_setup(Object *obj, VFIODevice *vbasedev, VFIORegion *region,

                      int index, const char *name)

{

    struct vfio_region_info *info;

    int ret;



    ret = vfio_get_region_info(vbasedev, index, &info);

    if (ret) {

        return ret;

    }



    region->vbasedev = vbasedev;

    region->flags = info->flags;

    region->size = info->size;

    region->fd_offset = info->offset;

    region->nr = index;



    if (region->size) {

        region->mem = g_new0(MemoryRegion, 1);

        memory_region_init_io(region->mem, obj, &vfio_region_ops,

                              region, name, region->size);



        if (!vbasedev->no_mmap &&

            region->flags & VFIO_REGION_INFO_FLAG_MMAP &&

            !(region->size & ~qemu_real_host_page_mask)) {



            ret = vfio_setup_region_sparse_mmaps(region, info);



            if (ret) {

                region->nr_mmaps = 1;

                region->mmaps = g_new0(VFIOMmap, region->nr_mmaps);

                region->mmaps[0].offset = 0;

                region->mmaps[0].size = region->size;

            }

        }

    }



    g_free(info);



    trace_vfio_region_setup(vbasedev->name, index, name,

                            region->flags, region->fd_offset, region->size);

    return 0;

}
