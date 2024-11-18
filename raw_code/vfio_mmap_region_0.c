int vfio_mmap_region(Object *obj, VFIORegion *region,

                     MemoryRegion *mem, MemoryRegion *submem,

                     void **map, size_t size, off_t offset,

                     const char *name)

{

    int ret = 0;

    VFIODevice *vbasedev = region->vbasedev;



    if (vbasedev->allow_mmap && size && region->flags &

        VFIO_REGION_INFO_FLAG_MMAP) {

        int prot = 0;



        if (region->flags & VFIO_REGION_INFO_FLAG_READ) {

            prot |= PROT_READ;

        }



        if (region->flags & VFIO_REGION_INFO_FLAG_WRITE) {

            prot |= PROT_WRITE;

        }



        *map = mmap(NULL, size, prot, MAP_SHARED,

                    vbasedev->fd,

                    region->fd_offset + offset);

        if (*map == MAP_FAILED) {

            *map = NULL;

            ret = -errno;

            goto empty_region;

        }



        memory_region_init_ram_ptr(submem, obj, name, size, *map);

        memory_region_set_skip_dump(submem);

    } else {

empty_region:

        /* Create a zero sized sub-region to make cleanup easy. */

        memory_region_init(submem, obj, name, 0);

    }



    memory_region_add_subregion(mem, offset, submem);



    return ret;

}
