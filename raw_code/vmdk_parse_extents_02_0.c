static int vmdk_parse_extents(const char *desc, BlockDriverState *bs,

                              const char *desc_file_path, QDict *options,

                              Error **errp)

{

    int ret;

    int matches;

    char access[11];

    char type[11];

    char fname[512];

    const char *p = desc;

    int64_t sectors = 0;

    int64_t flat_offset;

    char *extent_path;

    BdrvChild *extent_file;

    BDRVVmdkState *s = bs->opaque;

    VmdkExtent *extent;

    char extent_opt_prefix[32];

    Error *local_err = NULL;



    while (*p) {

        /* parse extent line in one of below formats:

         *

         * RW [size in sectors] FLAT "file-name.vmdk" OFFSET

         * RW [size in sectors] SPARSE "file-name.vmdk"

         * RW [size in sectors] VMFS "file-name.vmdk"

         * RW [size in sectors] VMFSSPARSE "file-name.vmdk"

         */

        flat_offset = -1;

        matches = sscanf(p, "%10s %" SCNd64 " %10s \"%511[^\n\r\"]\" %" SCNd64,

                         access, &sectors, type, fname, &flat_offset);

        if (matches < 4 || strcmp(access, "RW")) {

            goto next_line;

        } else if (!strcmp(type, "FLAT")) {

            if (matches != 5 || flat_offset < 0) {

                error_setg(errp, "Invalid extent lines: \n%s", p);

                return -EINVAL;

            }

        } else if (!strcmp(type, "VMFS")) {

            if (matches == 4) {

                flat_offset = 0;

            } else {

                error_setg(errp, "Invalid extent lines:\n%s", p);

                return -EINVAL;

            }

        } else if (matches != 4) {

            error_setg(errp, "Invalid extent lines:\n%s", p);

            return -EINVAL;

        }



        if (sectors <= 0 ||

            (strcmp(type, "FLAT") && strcmp(type, "SPARSE") &&

             strcmp(type, "VMFS") && strcmp(type, "VMFSSPARSE")) ||

            (strcmp(access, "RW"))) {

            goto next_line;

        }



        if (!path_is_absolute(fname) && !path_has_protocol(fname) &&

            !desc_file_path[0])

        {

            error_setg(errp, "Cannot use relative extent paths with VMDK "

                       "descriptor file '%s'", bs->file->bs->filename);

            return -EINVAL;

        }



        extent_path = g_malloc0(PATH_MAX);

        path_combine(extent_path, PATH_MAX, desc_file_path, fname);



        ret = snprintf(extent_opt_prefix, 32, "extents.%d", s->num_extents);

        assert(ret < 32);



        extent_file = bdrv_open_child(extent_path, options, extent_opt_prefix,

                                      bs, &child_file, false, &local_err);

        g_free(extent_path);

        if (local_err) {

            error_propagate(errp, local_err);

            return -EINVAL;

        }



        /* save to extents array */

        if (!strcmp(type, "FLAT") || !strcmp(type, "VMFS")) {

            /* FLAT extent */



            ret = vmdk_add_extent(bs, extent_file, true, sectors,

                            0, 0, 0, 0, 0, &extent, errp);

            if (ret < 0) {

                bdrv_unref_child(bs, extent_file);

                return ret;

            }

            extent->flat_start_offset = flat_offset << 9;

        } else if (!strcmp(type, "SPARSE") || !strcmp(type, "VMFSSPARSE")) {

            /* SPARSE extent and VMFSSPARSE extent are both "COWD" sparse file*/

            char *buf = vmdk_read_desc(extent_file->bs, 0, errp);

            if (!buf) {

                ret = -EINVAL;

            } else {

                ret = vmdk_open_sparse(bs, extent_file, bs->open_flags, buf,

                                       options, errp);

            }

            g_free(buf);

            if (ret) {

                bdrv_unref_child(bs, extent_file);

                return ret;

            }

            extent = &s->extents[s->num_extents - 1];

        } else {

            error_setg(errp, "Unsupported extent type '%s'", type);

            bdrv_unref_child(bs, extent_file);

            return -ENOTSUP;

        }

        extent->type = g_strdup(type);

next_line:

        /* move to next line */

        while (*p) {

            if (*p == '\n') {

                p++;

                break;

            }

            p++;

        }

    }

    return 0;

}
