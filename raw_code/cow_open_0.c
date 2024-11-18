static int cow_open(BlockDriverState *bs, QDict *options, int flags,

                    Error **errp)

{

    BDRVCowState *s = bs->opaque;

    struct cow_header_v2 cow_header;

    int bitmap_size;

    int64_t size;

    int ret;



    /* see if it is a cow image */

    ret = bdrv_pread(bs->file, 0, &cow_header, sizeof(cow_header));

    if (ret < 0) {

        goto fail;

    }



    if (be32_to_cpu(cow_header.magic) != COW_MAGIC) {

        error_setg(errp, "Image not in COW format");

        ret = -EINVAL;

        goto fail;

    }



    if (be32_to_cpu(cow_header.version) != COW_VERSION) {

        char version[64];

        snprintf(version, sizeof(version),

               "COW version %" PRIu32, cow_header.version);

        error_set(errp, QERR_UNKNOWN_BLOCK_FORMAT_FEATURE,

            bs->device_name, "cow", version);

        ret = -ENOTSUP;

        goto fail;

    }



    /* cow image found */

    size = be64_to_cpu(cow_header.size);

    bs->total_sectors = size / 512;



    pstrcpy(bs->backing_file, sizeof(bs->backing_file),

            cow_header.backing_file);



    bitmap_size = ((bs->total_sectors + 7) >> 3) + sizeof(cow_header);

    s->cow_sectors_offset = (bitmap_size + 511) & ~511;

    qemu_co_mutex_init(&s->lock);

    return 0;

 fail:

    return ret;

}
