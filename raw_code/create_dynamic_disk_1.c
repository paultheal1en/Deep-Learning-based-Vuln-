static int create_dynamic_disk(BlockBackend *blk, uint8_t *buf,

                               int64_t total_sectors)

{

    VHDDynDiskHeader *dyndisk_header =

        (VHDDynDiskHeader *) buf;

    size_t block_size, num_bat_entries;

    int i;

    int ret;

    int64_t offset = 0;



    // Write the footer (twice: at the beginning and at the end)

    block_size = 0x200000;

    num_bat_entries = (total_sectors + block_size / 512) / (block_size / 512);



    ret = blk_pwrite(blk, offset, buf, HEADER_SIZE);

    if (ret) {

        goto fail;

    }



    offset = 1536 + ((num_bat_entries * 4 + 511) & ~511);

    ret = blk_pwrite(blk, offset, buf, HEADER_SIZE);

    if (ret < 0) {

        goto fail;

    }



    // Write the initial BAT

    offset = 3 * 512;



    memset(buf, 0xFF, 512);

    for (i = 0; i < (num_bat_entries * 4 + 511) / 512; i++) {

        ret = blk_pwrite(blk, offset, buf, 512);

        if (ret < 0) {

            goto fail;

        }

        offset += 512;

    }



    // Prepare the Dynamic Disk Header

    memset(buf, 0, 1024);



    memcpy(dyndisk_header->magic, "cxsparse", 8);



    /*

     * Note: The spec is actually wrong here for data_offset, it says

     * 0xFFFFFFFF, but MS tools expect all 64 bits to be set.

     */

    dyndisk_header->data_offset = cpu_to_be64(0xFFFFFFFFFFFFFFFFULL);

    dyndisk_header->table_offset = cpu_to_be64(3 * 512);

    dyndisk_header->version = cpu_to_be32(0x00010000);

    dyndisk_header->block_size = cpu_to_be32(block_size);

    dyndisk_header->max_table_entries = cpu_to_be32(num_bat_entries);



    dyndisk_header->checksum = cpu_to_be32(vpc_checksum(buf, 1024));



    // Write the header

    offset = 512;



    ret = blk_pwrite(blk, offset, buf, 1024);

    if (ret < 0) {

        goto fail;

    }



 fail:

    return ret;

}
