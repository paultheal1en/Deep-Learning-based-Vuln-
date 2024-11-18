static int vhdx_open(BlockDriverState *bs, QDict *options, int flags,

                     Error **errp)

{

    BDRVVHDXState *s = bs->opaque;

    int ret = 0;

    uint32_t i;

    uint64_t signature;

    uint32_t data_blocks_cnt, bitmap_blocks_cnt;





    s->bat = NULL;

    s->first_visible_write = true;



    qemu_co_mutex_init(&s->lock);



    /* validate the file signature */

    ret = bdrv_pread(bs->file, 0, &signature, sizeof(uint64_t));

    if (ret < 0) {

        goto fail;

    }

    if (memcmp(&signature, "vhdxfile", 8)) {

        ret = -EINVAL;

        goto fail;

    }



    /* This is used for any header updates, for the file_write_guid.

     * The spec dictates that a new value should be used for the first

     * header update */

    vhdx_guid_generate(&s->session_guid);



    ret = vhdx_parse_header(bs, s);

    if (ret) {

        goto fail;

    }



    ret = vhdx_parse_log(bs, s);

    if (ret) {

        goto fail;

    }



    ret = vhdx_open_region_tables(bs, s);

    if (ret) {

        goto fail;

    }



    ret = vhdx_parse_metadata(bs, s);

    if (ret) {

        goto fail;

    }

    s->block_size = s->params.block_size;



    /* the VHDX spec dictates that virtual_disk_size is always a multiple of

     * logical_sector_size */

    bs->total_sectors = s->virtual_disk_size >> s->logical_sector_size_bits;



    data_blocks_cnt = s->virtual_disk_size >> s->block_size_bits;

    if (s->virtual_disk_size - (data_blocks_cnt << s->block_size_bits)) {

        data_blocks_cnt++;

    }

    bitmap_blocks_cnt = data_blocks_cnt >> s->chunk_ratio_bits;

    if (data_blocks_cnt - (bitmap_blocks_cnt << s->chunk_ratio_bits)) {

        bitmap_blocks_cnt++;

    }



    if (s->parent_entries) {

        s->bat_entries = bitmap_blocks_cnt * (s->chunk_ratio + 1);

    } else {

        s->bat_entries = data_blocks_cnt +

                         ((data_blocks_cnt - 1) >> s->chunk_ratio_bits);

    }



    s->bat_offset = s->bat_rt.file_offset;



    if (s->bat_entries > s->bat_rt.length / sizeof(VHDXBatEntry)) {

        /* BAT allocation is not large enough for all entries */

        ret = -EINVAL;

        goto fail;

    }



    /* s->bat is freed in vhdx_close() */

    s->bat = qemu_blockalign(bs, s->bat_rt.length);



    ret = bdrv_pread(bs->file, s->bat_offset, s->bat, s->bat_rt.length);

    if (ret < 0) {

        goto fail;

    }



    for (i = 0; i < s->bat_entries; i++) {

        le64_to_cpus(&s->bat[i]);

    }



    if (flags & BDRV_O_RDWR) {

        ret = vhdx_update_headers(bs, s, false, NULL);

        if (ret < 0) {

            goto fail;

        }

    }



    /* TODO: differencing files, write */



    /* Disable migration when VHDX images are used */

    error_set(&s->migration_blocker,

            QERR_BLOCK_FORMAT_FEATURE_NOT_SUPPORTED,

            "vhdx", bs->device_name, "live migration");

    migrate_add_blocker(s->migration_blocker);



    return 0;

fail:

    qemu_vfree(s->headers[0]);

    qemu_vfree(s->headers[1]);

    qemu_vfree(s->bat);

    qemu_vfree(s->parent_entries);

    return ret;

}
