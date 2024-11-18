static int vmdk_create(const char *filename, QEMUOptionParameter *options,

                       Error **errp)

{

    int fd, idx = 0;

    char desc[BUF_SIZE];

    int64_t total_size = 0, filesize;

    const char *adapter_type = NULL;

    const char *backing_file = NULL;

    const char *fmt = NULL;

    int flags = 0;

    int ret = 0;

    bool flat, split, compress;

    char ext_desc_lines[BUF_SIZE] = "";

    char path[PATH_MAX], prefix[PATH_MAX], postfix[PATH_MAX];

    const int64_t split_size = 0x80000000;  /* VMDK has constant split size */

    const char *desc_extent_line;

    char parent_desc_line[BUF_SIZE] = "";

    uint32_t parent_cid = 0xffffffff;

    uint32_t number_heads = 16;

    bool zeroed_grain = false;

    const char desc_template[] =

        "# Disk DescriptorFile\n"

        "version=1\n"

        "CID=%x\n"

        "parentCID=%x\n"

        "createType=\"%s\"\n"

        "%s"

        "\n"

        "# Extent description\n"

        "%s"

        "\n"

        "# The Disk Data Base\n"

        "#DDB\n"

        "\n"

        "ddb.virtualHWVersion = \"%d\"\n"

        "ddb.geometry.cylinders = \"%" PRId64 "\"\n"

        "ddb.geometry.heads = \"%d\"\n"

        "ddb.geometry.sectors = \"63\"\n"

        "ddb.adapterType = \"%s\"\n";



    if (filename_decompose(filename, path, prefix, postfix, PATH_MAX, errp)) {

        return -EINVAL;

    }

    /* Read out options */

    while (options && options->name) {

        if (!strcmp(options->name, BLOCK_OPT_SIZE)) {

            total_size = options->value.n;

        } else if (!strcmp(options->name, BLOCK_OPT_ADAPTER_TYPE)) {

            adapter_type = options->value.s;

        } else if (!strcmp(options->name, BLOCK_OPT_BACKING_FILE)) {

            backing_file = options->value.s;

        } else if (!strcmp(options->name, BLOCK_OPT_COMPAT6)) {

            flags |= options->value.n ? BLOCK_FLAG_COMPAT6 : 0;

        } else if (!strcmp(options->name, BLOCK_OPT_SUBFMT)) {

            fmt = options->value.s;

        } else if (!strcmp(options->name, BLOCK_OPT_ZEROED_GRAIN)) {

            zeroed_grain |= options->value.n;

        }

        options++;

    }

    if (!adapter_type) {

        adapter_type = "ide";

    } else if (strcmp(adapter_type, "ide") &&

               strcmp(adapter_type, "buslogic") &&

               strcmp(adapter_type, "lsilogic") &&

               strcmp(adapter_type, "legacyESX")) {

        error_setg(errp, "Unknown adapter type: '%s'", adapter_type);

        return -EINVAL;

    }

    if (strcmp(adapter_type, "ide") != 0) {

        /* that's the number of heads with which vmware operates when

           creating, exporting, etc. vmdk files with a non-ide adapter type */

        number_heads = 255;

    }

    if (!fmt) {

        /* Default format to monolithicSparse */

        fmt = "monolithicSparse";

    } else if (strcmp(fmt, "monolithicFlat") &&

               strcmp(fmt, "monolithicSparse") &&

               strcmp(fmt, "twoGbMaxExtentSparse") &&

               strcmp(fmt, "twoGbMaxExtentFlat") &&

               strcmp(fmt, "streamOptimized")) {

        error_setg(errp, "Unknown subformat: '%s'", fmt);

        return -EINVAL;

    }

    split = !(strcmp(fmt, "twoGbMaxExtentFlat") &&

              strcmp(fmt, "twoGbMaxExtentSparse"));

    flat = !(strcmp(fmt, "monolithicFlat") &&

             strcmp(fmt, "twoGbMaxExtentFlat"));

    compress = !strcmp(fmt, "streamOptimized");

    if (flat) {

        desc_extent_line = "RW %lld FLAT \"%s\" 0\n";

    } else {

        desc_extent_line = "RW %lld SPARSE \"%s\"\n";

    }

    if (flat && backing_file) {

        error_setg(errp, "Flat image can't have backing file");

        return -ENOTSUP;

    }

    if (flat && zeroed_grain) {

        error_setg(errp, "Flat image can't enable zeroed grain");

        return -ENOTSUP;

    }

    if (backing_file) {

        BlockDriverState *bs = bdrv_new("");

        ret = bdrv_open(bs, backing_file, NULL, 0, NULL, errp);

        if (ret != 0) {

            bdrv_unref(bs);

            return ret;

        }

        if (strcmp(bs->drv->format_name, "vmdk")) {

            bdrv_unref(bs);

            return -EINVAL;

        }

        parent_cid = vmdk_read_cid(bs, 0);

        bdrv_unref(bs);

        snprintf(parent_desc_line, sizeof(parent_desc_line),

                "parentFileNameHint=\"%s\"", backing_file);

    }



    /* Create extents */

    filesize = total_size;

    while (filesize > 0) {

        char desc_line[BUF_SIZE];

        char ext_filename[PATH_MAX];

        char desc_filename[PATH_MAX];

        int64_t size = filesize;



        if (split && size > split_size) {

            size = split_size;

        }

        if (split) {

            snprintf(desc_filename, sizeof(desc_filename), "%s-%c%03d%s",

                    prefix, flat ? 'f' : 's', ++idx, postfix);

        } else if (flat) {

            snprintf(desc_filename, sizeof(desc_filename), "%s-flat%s",

                    prefix, postfix);

        } else {

            snprintf(desc_filename, sizeof(desc_filename), "%s%s",

                    prefix, postfix);

        }

        snprintf(ext_filename, sizeof(ext_filename), "%s%s",

                path, desc_filename);



        if (vmdk_create_extent(ext_filename, size,

                               flat, compress, zeroed_grain)) {

            return -EINVAL;

        }

        filesize -= size;



        /* Format description line */

        snprintf(desc_line, sizeof(desc_line),

                    desc_extent_line, size / 512, desc_filename);

        pstrcat(ext_desc_lines, sizeof(ext_desc_lines), desc_line);

    }

    /* generate descriptor file */

    snprintf(desc, sizeof(desc), desc_template,

            (unsigned int)time(NULL),

            parent_cid,

            fmt,

            parent_desc_line,

            ext_desc_lines,

            (flags & BLOCK_FLAG_COMPAT6 ? 6 : 4),

            total_size / (int64_t)(63 * number_heads * 512), number_heads,

                adapter_type);

    if (split || flat) {

        fd = qemu_open(filename,

                       O_WRONLY | O_CREAT | O_TRUNC | O_BINARY | O_LARGEFILE,

                       0644);

    } else {

        fd = qemu_open(filename,

                       O_WRONLY | O_BINARY | O_LARGEFILE,

                       0644);

    }

    if (fd < 0) {

        return -errno;

    }

    /* the descriptor offset = 0x200 */

    if (!split && !flat && 0x200 != lseek(fd, 0x200, SEEK_SET)) {

        ret = -errno;

        goto exit;

    }

    ret = qemu_write_full(fd, desc, strlen(desc));

    if (ret != strlen(desc)) {

        ret = -errno;

        goto exit;

    }

    ret = 0;

exit:

    qemu_close(fd);

    return ret;

}
