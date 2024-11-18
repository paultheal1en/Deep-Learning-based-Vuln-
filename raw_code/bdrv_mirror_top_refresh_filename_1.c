static void bdrv_mirror_top_refresh_filename(BlockDriverState *bs, QDict *opts)

{






    bdrv_refresh_filename(bs->backing->bs);

    pstrcpy(bs->exact_filename, sizeof(bs->exact_filename),

            bs->backing->bs->filename);
