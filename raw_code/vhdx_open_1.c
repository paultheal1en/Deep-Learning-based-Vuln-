static int vhdx_open(BlockDriverState *bs, QDict *options, int flags,

                     Error **errp)

{

    BDRVVHDXState *s = bs->opaque;

    int ret = 0;

    uint32_t i;

    uint64_t signature;

    bool log_flushed = false;





    s->bat = NULL;

    s->first_visible_write = true;



    qemu_co_mutex_init(&s->lock);

    QLIST_INIT(&s->regions);



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

    if (ret < 0) {

        goto fail;

    }



    ret = vhdx_parse_log(bs, s, &log_flushed);

    if (ret < 0) {

        goto fail;

    }



    ret = vhdx_open_region_tables(bs, s);

    if (ret < 0) {

        goto fail;

    }



    ret = vhdx_parse_metadata(bs, s);

    if (ret < 0) {

        goto fail;

    }



    s->block_size = s->params.block_size;



    /* the VHDX spec dictates that virtual_disk_size is always a multiple of

     * logical_sector_size */

    bs->total_sectors = s->virtual_disk_size >> s->logical_sector_size_bits;



    vhdx_calc_bat_entries(s);



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



    uint64_t payblocks = s->chunk_ratio;

    /* endian convert, and verify populated BAT field file offsets against

     * region table and log entries */

    for (i = 0; i < s->bat_entries; i++) {

        le64_to_cpus(&s->bat[i]);

        if (payblocks--) {

            /* payload bat entries */

            if ((s->bat[i] & VHDX_BAT_STATE_BIT_MASK) ==

                    PAYLOAD_BLOCK_FULLY_PRESENT) {

                ret = vhdx_region_check(s, s->bat[i] & VHDX_BAT_FILE_OFF_MASK,

                                        s->block_size);

                if (ret < 0) {

                    goto fail;

                }

            }

        } else {

            payblocks = s->chunk_ratio;

            /* Once differencing files are supported, verify sector bitmap

             * blocks here */

        }

    }



    if (flags & BDRV_O_RDWR) {

        ret = vhdx_update_headers(bs, s, false, NULL);

        if (ret < 0) {

            goto fail;

        }

    }



    /* TODO: differencing files */



    /* Disable migration when VHDX images are used */

    error_set(&s->migration_blocker,

            QERR_BLOCK_FORMAT_FEATURE_NOT_SUPPORTED,

            "vhdx", bs->device_name, "live migration");

    migrate_add_blocker(s->migration_blocker);



    return 0;

fail:

    vhdx_close(bs);

    return ret;

}
