int bdrv_open_backing_file(BlockDriverState *bs)

{

    char backing_filename[PATH_MAX];

    int back_flags, ret;

    BlockDriver *back_drv = NULL;



    if (bs->backing_hd != NULL) {

        return 0;

    }



    bs->open_flags &= ~BDRV_O_NO_BACKING;

    if (bs->backing_file[0] == '\0') {

        return 0;

    }



    bs->backing_hd = bdrv_new("");

    bdrv_get_full_backing_filename(bs, backing_filename,

                                   sizeof(backing_filename));



    if (bs->backing_format[0] != '\0') {

        back_drv = bdrv_find_format(bs->backing_format);

    }



    /* backing files always opened read-only */

    back_flags = bs->open_flags & ~(BDRV_O_RDWR | BDRV_O_SNAPSHOT);



    ret = bdrv_open(bs->backing_hd, backing_filename, NULL,

                    back_flags, back_drv);

    if (ret < 0) {

        bdrv_delete(bs->backing_hd);

        bs->backing_hd = NULL;

        bs->open_flags |= BDRV_O_NO_BACKING;

        return ret;

    }

    return 0;

}
