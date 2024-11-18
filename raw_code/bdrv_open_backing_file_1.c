int bdrv_open_backing_file(BlockDriverState *bs, QDict *options, Error **errp)

{

    char backing_filename[PATH_MAX];

    int back_flags, ret;

    BlockDriver *back_drv = NULL;

    Error *local_err = NULL;



    if (bs->backing_hd != NULL) {

        QDECREF(options);

        return 0;

    }



    /* NULL means an empty set of options */

    if (options == NULL) {

        options = qdict_new();

    }



    bs->open_flags &= ~BDRV_O_NO_BACKING;

    if (qdict_haskey(options, "file.filename")) {

        backing_filename[0] = '\0';

    } else if (bs->backing_file[0] == '\0' && qdict_size(options) == 0) {

        QDECREF(options);

        return 0;

    } else {

        bdrv_get_full_backing_filename(bs, backing_filename,

                                       sizeof(backing_filename));

    }



    bs->backing_hd = bdrv_new("");



    if (bs->backing_format[0] != '\0') {

        back_drv = bdrv_find_format(bs->backing_format);

    }



    /* backing files always opened read-only */

    back_flags = bs->open_flags & ~(BDRV_O_RDWR | BDRV_O_SNAPSHOT);



    ret = bdrv_open(bs->backing_hd,

                    *backing_filename ? backing_filename : NULL, options,

                    back_flags, back_drv, &local_err);

    pstrcpy(bs->backing_file, sizeof(bs->backing_file),

            bs->backing_hd->file->filename);

    if (ret < 0) {

        bdrv_unref(bs->backing_hd);

        bs->backing_hd = NULL;

        bs->open_flags |= BDRV_O_NO_BACKING;

        error_propagate(errp, local_err);

        return ret;

    }

    return 0;

}
