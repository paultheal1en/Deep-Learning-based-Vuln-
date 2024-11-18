bool virtio_disk_is_eckd(void)

{

    if (guessed_disk_nature) {

        return (blk_cfg.blk_size  == 4096);

    }

    return (blk_cfg.geometry.heads == 15)

        && (blk_cfg.geometry.sectors == 12)

        && (blk_cfg.blk_size  == 4096);

}
