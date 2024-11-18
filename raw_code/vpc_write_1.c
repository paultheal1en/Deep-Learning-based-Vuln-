static int vpc_write(BlockDriverState *bs, int64_t sector_num,

    const uint8_t *buf, int nb_sectors)

{

    BDRVVPCState *s = bs->opaque;

    int64_t offset;

    int64_t sectors, sectors_per_block;

    int ret;

    VHDFooter *footer =  (VHDFooter *) s->footer_buf;



    if (cpu_to_be32(footer->type) == VHD_FIXED) {

        return bdrv_write(bs->file, sector_num, buf, nb_sectors);

    }

    while (nb_sectors > 0) {

        offset = get_sector_offset(bs, sector_num, 1);



        sectors_per_block = s->block_size >> BDRV_SECTOR_BITS;

        sectors = sectors_per_block - (sector_num % sectors_per_block);

        if (sectors > nb_sectors) {

            sectors = nb_sectors;

        }



        if (offset == -1) {

            offset = alloc_block(bs, sector_num);

            if (offset < 0)

                return -1;

        }



        ret = bdrv_pwrite(bs->file, offset, buf, sectors * BDRV_SECTOR_SIZE);

        if (ret != sectors * BDRV_SECTOR_SIZE) {

            return -1;

        }



        nb_sectors -= sectors;

        sector_num += sectors;

        buf += sectors * BDRV_SECTOR_SIZE;

    }



    return 0;

}
