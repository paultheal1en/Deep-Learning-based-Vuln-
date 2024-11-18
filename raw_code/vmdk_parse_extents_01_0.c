static int vmdk_parse_extents(const char *desc, BlockDriverState *bs,

        const char *desc_file_path)

{

    int ret;

    char access[11];

    char type[11];

    char fname[512];

    const char *p = desc;

    int64_t sectors = 0;

    int64_t flat_offset;

    char extent_path[PATH_MAX];

    BlockDriverState *extent_file;



    while (*p) {

        /* parse extent line:

         * RW [size in sectors] FLAT "file-name.vmdk" OFFSET

         * or

         * RW [size in sectors] SPARSE "file-name.vmdk"

         */

        flat_offset = -1;

        ret = sscanf(p, "%10s %" SCNd64 " %10s \"%511[^\n\r\"]\" %" SCNd64,

                access, &sectors, type, fname, &flat_offset);

        if (ret < 4 || strcmp(access, "RW")) {

            goto next_line;

        } else if (!strcmp(type, "FLAT")) {

            if (ret != 5 || flat_offset < 0) {

                return -EINVAL;

            }

        } else if (ret != 4) {

            return -EINVAL;

        }



        if (sectors <= 0 ||

            (strcmp(type, "FLAT") && strcmp(type, "SPARSE")) ||

            (strcmp(access, "RW"))) {

            goto next_line;

        }



        path_combine(extent_path, sizeof(extent_path),

                desc_file_path, fname);

        ret = bdrv_file_open(&extent_file, extent_path, NULL, bs->open_flags);

        if (ret) {

            return ret;

        }



        /* save to extents array */

        if (!strcmp(type, "FLAT")) {

            /* FLAT extent */

            VmdkExtent *extent;



            extent = vmdk_add_extent(bs, extent_file, true, sectors,

                            0, 0, 0, 0, sectors);

            extent->flat_start_offset = flat_offset << 9;

        } else if (!strcmp(type, "SPARSE")) {

            /* SPARSE extent */

            ret = vmdk_open_sparse(bs, extent_file, bs->open_flags);

            if (ret) {

                bdrv_delete(extent_file);

                return ret;

            }

        } else {

            fprintf(stderr,

                "VMDK: Not supported extent type \"%s\""".\n", type);

            return -ENOTSUP;

        }

next_line:

        /* move to next line */

        while (*p && *p != '\n') {

            p++;

        }

        p++;

    }

    return 0;

}
