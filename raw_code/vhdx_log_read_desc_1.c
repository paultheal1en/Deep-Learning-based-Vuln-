static int vhdx_log_read_desc(BlockDriverState *bs, BDRVVHDXState *s,

                              VHDXLogEntries *log, VHDXLogDescEntries **buffer,

                              bool convert_endian)

{

    int ret = 0;

    uint32_t desc_sectors;

    uint32_t sectors_read;

    VHDXLogEntryHeader hdr;

    VHDXLogDescEntries *desc_entries = NULL;

    VHDXLogDescriptor desc;

    int i;



    assert(*buffer == NULL);



    ret = vhdx_log_peek_hdr(bs, log, &hdr);

    if (ret < 0) {

        goto exit;

    }



    if (vhdx_log_hdr_is_valid(log, &hdr, s) == false) {

        ret = -EINVAL;

        goto exit;

    }



    desc_sectors = vhdx_compute_desc_sectors(hdr.descriptor_count);

    desc_entries = qemu_blockalign(bs, desc_sectors * VHDX_LOG_SECTOR_SIZE);



    ret = vhdx_log_read_sectors(bs, log, &sectors_read, desc_entries,

                                desc_sectors, false);

    if (ret < 0) {

        goto free_and_exit;

    }

    if (sectors_read != desc_sectors) {

        ret = -EINVAL;

        goto free_and_exit;

    }



    /* put in proper endianness, and validate each desc */

    for (i = 0; i < hdr.descriptor_count; i++) {

        desc = desc_entries->desc[i];

        vhdx_log_desc_le_import(&desc);

        if (convert_endian) {

            desc_entries->desc[i] = desc;

        }

        if (vhdx_log_desc_is_valid(&desc, &hdr) == false) {

            ret = -EINVAL;

            goto free_and_exit;

        }

    }

    if (convert_endian) {

        desc_entries->hdr = hdr;

    }



    *buffer = desc_entries;

    goto exit;



free_and_exit:

    qemu_vfree(desc_entries);

exit:

    return ret;

}
