static bool virtio_blk_sect_range_ok(VirtIOBlock *dev,

                                     uint64_t sector, size_t size)

{

    uint64_t nb_sectors = size >> BDRV_SECTOR_BITS;

    uint64_t total_sectors;



    if (nb_sectors > INT_MAX) {

        return false;

    }

    if (sector & dev->sector_mask) {

        return false;

    }

    if (size % dev->conf.conf.logical_block_size) {

        return false;

    }

    blk_get_geometry(dev->blk, &total_sectors);

    if (sector > total_sectors || nb_sectors > total_sectors - sector) {

        return false;

    }

    return true;

}
