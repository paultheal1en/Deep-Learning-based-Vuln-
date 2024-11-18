static int vhdx_create(const char *filename, QEMUOptionParameter *options,

                       Error **errp)

{

    int ret = 0;

    uint64_t image_size = (uint64_t) 2 * GiB;

    uint32_t log_size   = 1 * MiB;

    uint32_t block_size = 0;

    uint64_t signature;

    uint64_t metadata_offset;

    bool use_zero_blocks = false;



    gunichar2 *creator = NULL;

    glong creator_items;

    BlockDriverState *bs;

    const char *type = NULL;

    VHDXImageType image_type;

    Error *local_err = NULL;



    while (options && options->name) {

        if (!strcmp(options->name, BLOCK_OPT_SIZE)) {

            image_size = options->value.n;

        } else if (!strcmp(options->name, VHDX_BLOCK_OPT_LOG_SIZE)) {

            log_size = options->value.n;

        } else if (!strcmp(options->name, VHDX_BLOCK_OPT_BLOCK_SIZE)) {

            block_size = options->value.n;

        } else if (!strcmp(options->name, BLOCK_OPT_SUBFMT)) {

            type = options->value.s;

        } else if (!strcmp(options->name, VHDX_BLOCK_OPT_ZERO)) {

            use_zero_blocks = options->value.n != 0;

        }

        options++;

    }



    if (image_size > VHDX_MAX_IMAGE_SIZE) {

        error_setg_errno(errp, EINVAL, "Image size too large; max of 64TB");

        ret = -EINVAL;

        goto exit;

    }



    if (type == NULL) {

        type = "dynamic";

    }



    if (!strcmp(type, "dynamic")) {

        image_type = VHDX_TYPE_DYNAMIC;

    } else if (!strcmp(type, "fixed")) {

        image_type = VHDX_TYPE_FIXED;

    } else if (!strcmp(type, "differencing")) {

        error_setg_errno(errp, ENOTSUP,

                         "Differencing files not yet supported");

        ret = -ENOTSUP;

        goto exit;

    } else {

        ret = -EINVAL;

        goto exit;

    }



    /* These are pretty arbitrary, and mainly designed to keep the BAT

     * size reasonable to load into RAM */

    if (block_size == 0) {

        if (image_size > 32 * TiB) {

            block_size = 64 * MiB;

        } else if (image_size > (uint64_t) 100 * GiB) {

            block_size = 32 * MiB;

        } else if (image_size > 1 * GiB) {

            block_size = 16 * MiB;

        } else {

            block_size = 8 * MiB;

        }

    }





    /* make the log size close to what was specified, but must be

     * min 1MB, and multiple of 1MB */

    log_size = ROUND_UP(log_size, MiB);



    block_size = ROUND_UP(block_size, MiB);

    block_size = block_size > VHDX_BLOCK_SIZE_MAX ? VHDX_BLOCK_SIZE_MAX :

                                                    block_size;



    ret = bdrv_create_file(filename, options, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto exit;

    }



    ret = bdrv_file_open(&bs, filename, NULL, NULL, BDRV_O_RDWR, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto exit;

    }



    /* Create (A) */



    /* The creator field is optional, but may be useful for

     * debugging / diagnostics */

    creator = g_utf8_to_utf16("QEMU v" QEMU_VERSION, -1, NULL,

                              &creator_items, NULL);

    signature = cpu_to_le64(VHDX_FILE_SIGNATURE);

    bdrv_pwrite(bs, VHDX_FILE_ID_OFFSET, &signature, sizeof(signature));

    if (ret < 0) {

        goto delete_and_exit;

    }

    if (creator) {

        bdrv_pwrite(bs, VHDX_FILE_ID_OFFSET + sizeof(signature), creator,

                    creator_items * sizeof(gunichar2));

        if (ret < 0) {

            goto delete_and_exit;

        }

    }





    /* Creates (B),(C) */

    ret = vhdx_create_new_headers(bs, image_size, log_size);

    if (ret < 0) {

        goto delete_and_exit;

    }



    /* Creates (D),(E),(G) explicitly. (F) created as by-product */

    ret = vhdx_create_new_region_table(bs, image_size, block_size, 512,

                                       log_size, use_zero_blocks, image_type,

                                       &metadata_offset);

    if (ret < 0) {

        goto delete_and_exit;

    }



    /* Creates (H) */

    ret = vhdx_create_new_metadata(bs, image_size, block_size, 512,

                                   metadata_offset, image_type);

    if (ret < 0) {

        goto delete_and_exit;

    }







delete_and_exit:

    bdrv_unref(bs);

exit:

    g_free(creator);

    return ret;

}
