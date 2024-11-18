void bdrv_set_boot_sector(BlockDriverState *bs, const uint8_t *data, int size)

{

    bs->boot_sector_enabled = 1;

    if (size > 512)

        size = 512;

    memcpy(bs->boot_sector_data, data, size);

    memset(bs->boot_sector_data + size, 0, 512 - size);

}
