static bool virtio_blk_sect_range_ok(VirtIOBlock *dev,
                                     uint64_t sector, size_t size)
{
    if (sector & dev->sector_mask) {
    if (size % dev->conf->logical_block_size) {
    return true;