static int check_empty_sectors(BlockBackend *blk, int64_t sect_num,

                               int sect_count, const char *filename,

                               uint8_t *buffer, bool quiet)

{

    int pnum, ret = 0;

    ret = blk_pread(blk, sect_num << BDRV_SECTOR_BITS, buffer,

                    sect_count << BDRV_SECTOR_BITS);

    if (ret < 0) {

        error_report("Error while reading offset %" PRId64 " of %s: %s",

                     sectors_to_bytes(sect_num), filename, strerror(-ret));

        return ret;

    }

    ret = is_allocated_sectors(buffer, sect_count, &pnum);

    if (ret || pnum != sect_count) {

        qprintf(quiet, "Content mismatch at offset %" PRId64 "!\n",

                sectors_to_bytes(ret ? sect_num : sect_num + pnum));

        return 1;

    }



    return 0;

}
