static int vmdk_create(const char *filename, QemuOpts *opts, Error **errp)

{

    int idx = 0;

    BlockBackend *new_blk = NULL;

    Error *local_err = NULL;

    char *desc = NULL;

    int64_t total_size = 0, filesize;

    char *adapter_type = NULL;

    char *backing_file = NULL;

    char *hw_version = NULL;

    char *fmt = NULL;

    int ret = 0;

    bool flat, split, compress;

    GString *ext_desc_lines;

    char *path = g_malloc0(PATH_MAX);

    char *prefix = g_malloc0(PATH_MAX);

    char *postfix = g_malloc0(PATH_MAX);

    char *desc_line = g_malloc0(BUF_SIZE);

    char *ext_filename = g_malloc0(PATH_MAX);

    char *desc_filename = g_malloc0(PATH_MAX);

    const int64_t split_size = 0x80000000;  /* VMDK has constant split size */

    const char *desc_extent_line;

    char *parent_desc_line = g_malloc0(BUF_SIZE);

    uint32_t parent_cid = 0xffffffff;

    uint32_t number_heads = 16;

    bool zeroed_grain = false;

    uint32_t desc_offset = 0, desc_len;

    const char desc_template[] =

        "# Disk DescriptorFile\n"

        "version=1\n"

        "CID=%" PRIx32 "\n"

        "parentCID=%" PRIx32 "\n"

        "createType=\"%s\"\n"

        "%s"

        "\n"

        "# Extent description\n"

        "%s"

        "\n"

        "# The Disk Data Base\n"

        "#DDB\n"

        "\n"

        "ddb.virtualHWVersion = \"%s\"\n"

        "ddb.geometry.cylinders = \"%" PRId64 "\"\n"

        "ddb.geometry.heads = \"%" PRIu32 "\"\n"

        "ddb.geometry.sectors = \"63\"\n"

        "ddb.adapterType = \"%s\"\n";



    ext_desc_lines = g_string_new(NULL);



    if (filename_decompose(filename, path, prefix, postfix, PATH_MAX, errp)) {

        ret = -EINVAL;

        goto exit;

    }

    /* Read out options */

    total_size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),

                          BDRV_SECTOR_SIZE);

    adapter_type = qemu_opt_get_del(opts, BLOCK_OPT_ADAPTER_TYPE);

    backing_file = qemu_opt_get_del(opts, BLOCK_OPT_BACKING_FILE);

    hw_version = qemu_opt_get_del(opts, BLOCK_OPT_HWVERSION);

    if (qemu_opt_get_bool_del(opts, BLOCK_OPT_COMPAT6, false)) {

        if (strcmp(hw_version, "undefined")) {

            error_setg(errp,

                       "compat6 cannot be enabled with hwversion set");

            ret = -EINVAL;

            goto exit;

        }

        g_free(hw_version);

        hw_version = g_strdup("6");

    }

    if (strcmp(hw_version, "undefined") == 0) {

        g_free(hw_version);

        hw_version = g_strdup("4");

    }

    fmt = qemu_opt_get_del(opts, BLOCK_OPT_SUBFMT);

    if (qemu_opt_get_bool_del(opts, BLOCK_OPT_ZEROED_GRAIN, false)) {

        zeroed_grain = true;

    }



    if (!adapter_type) {

        adapter_type = g_strdup("ide");

    } else if (strcmp(adapter_type, "ide") &&

               strcmp(adapter_type, "buslogic") &&

               strcmp(adapter_type, "lsilogic") &&

               strcmp(adapter_type, "legacyESX")) {

        error_setg(errp, "Unknown adapter type: '%s'", adapter_type);

        ret = -EINVAL;

        goto exit;

    }

    if (strcmp(adapter_type, "ide") != 0) {

        /* that's the number of heads with which vmware operates when

           creating, exporting, etc. vmdk files with a non-ide adapter type */

        number_heads = 255;

    }

    if (!fmt) {

        /* Default format to monolithicSparse */

        fmt = g_strdup("monolithicSparse");

    } else if (strcmp(fmt, "monolithicFlat") &&

               strcmp(fmt, "monolithicSparse") &&

               strcmp(fmt, "twoGbMaxExtentSparse") &&

               strcmp(fmt, "twoGbMaxExtentFlat") &&

               strcmp(fmt, "streamOptimized")) {

        error_setg(errp, "Unknown subformat: '%s'", fmt);

        ret = -EINVAL;

        goto exit;

    }

    split = !(strcmp(fmt, "twoGbMaxExtentFlat") &&

              strcmp(fmt, "twoGbMaxExtentSparse"));

    flat = !(strcmp(fmt, "monolithicFlat") &&

             strcmp(fmt, "twoGbMaxExtentFlat"));

    compress = !strcmp(fmt, "streamOptimized");

    if (flat) {

        desc_extent_line = "RW %" PRId64 " FLAT \"%s\" 0\n";

    } else {

        desc_extent_line = "RW %" PRId64 " SPARSE \"%s\"\n";

    }

    if (flat && backing_file) {

        error_setg(errp, "Flat image can't have backing file");

        ret = -ENOTSUP;

        goto exit;

    }

    if (flat && zeroed_grain) {

        error_setg(errp, "Flat image can't enable zeroed grain");

        ret = -ENOTSUP;

        goto exit;

    }

    if (backing_file) {

        BlockBackend *blk;

        char *full_backing = g_new0(char, PATH_MAX);

        bdrv_get_full_backing_filename_from_filename(filename, backing_file,

                                                     full_backing, PATH_MAX,

                                                     &local_err);

        if (local_err) {

            g_free(full_backing);

            error_propagate(errp, local_err);

            ret = -ENOENT;

            goto exit;

        }



        blk = blk_new_open(full_backing, NULL, NULL,

                           BDRV_O_NO_BACKING, errp);

        g_free(full_backing);

        if (blk == NULL) {

            ret = -EIO;

            goto exit;

        }

        if (strcmp(blk_bs(blk)->drv->format_name, "vmdk")) {

            blk_unref(blk);

            ret = -EINVAL;

            goto exit;

        }

        parent_cid = vmdk_read_cid(blk_bs(blk), 0);

        blk_unref(blk);

        snprintf(parent_desc_line, BUF_SIZE,

                "parentFileNameHint=\"%s\"", backing_file);

    }



    /* Create extents */

    filesize = total_size;

    while (filesize > 0) {

        int64_t size = filesize;



        if (split && size > split_size) {

            size = split_size;

        }

        if (split) {

            snprintf(desc_filename, PATH_MAX, "%s-%c%03d%s",

                    prefix, flat ? 'f' : 's', ++idx, postfix);

        } else if (flat) {

            snprintf(desc_filename, PATH_MAX, "%s-flat%s", prefix, postfix);

        } else {

            snprintf(desc_filename, PATH_MAX, "%s%s", prefix, postfix);

        }

        snprintf(ext_filename, PATH_MAX, "%s%s", path, desc_filename);



        if (vmdk_create_extent(ext_filename, size,

                               flat, compress, zeroed_grain, opts, errp)) {

            ret = -EINVAL;

            goto exit;

        }

        filesize -= size;



        /* Format description line */

        snprintf(desc_line, BUF_SIZE,

                    desc_extent_line, size / BDRV_SECTOR_SIZE, desc_filename);

        g_string_append(ext_desc_lines, desc_line);

    }

    /* generate descriptor file */

    desc = g_strdup_printf(desc_template,

                           g_random_int(),

                           parent_cid,

                           fmt,

                           parent_desc_line,

                           ext_desc_lines->str,

                           hw_version,

                           total_size /

                               (int64_t)(63 * number_heads * BDRV_SECTOR_SIZE),

                           number_heads,

                           adapter_type);

    desc_len = strlen(desc);

    /* the descriptor offset = 0x200 */

    if (!split && !flat) {

        desc_offset = 0x200;

    } else {

        ret = bdrv_create_file(filename, opts, &local_err);

        if (ret < 0) {

            error_propagate(errp, local_err);

            goto exit;

        }

    }



    new_blk = blk_new_open(filename, NULL, NULL,

                           BDRV_O_RDWR | BDRV_O_RESIZE | BDRV_O_PROTOCOL,

                           &local_err);

    if (new_blk == NULL) {

        error_propagate(errp, local_err);

        ret = -EIO;

        goto exit;

    }



    blk_set_allow_write_beyond_eof(new_blk, true);



    ret = blk_pwrite(new_blk, desc_offset, desc, desc_len, 0);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not write description");

        goto exit;

    }

    /* bdrv_pwrite write padding zeros to align to sector, we don't need that

     * for description file */

    if (desc_offset == 0) {

        ret = blk_truncate(new_blk, desc_len, PREALLOC_MODE_OFF, errp);

    }

exit:

    if (new_blk) {

        blk_unref(new_blk);

    }

    g_free(adapter_type);

    g_free(backing_file);

    g_free(hw_version);

    g_free(fmt);

    g_free(desc);

    g_free(path);

    g_free(prefix);

    g_free(postfix);

    g_free(desc_line);

    g_free(ext_filename);

    g_free(desc_filename);

    g_free(parent_desc_line);

    g_string_free(ext_desc_lines, true);

    return ret;

}
