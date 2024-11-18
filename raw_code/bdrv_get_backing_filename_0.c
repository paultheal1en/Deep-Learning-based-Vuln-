void bdrv_get_backing_filename(BlockDriverState *bs,

                               char *filename, int filename_size)

{

    pstrcpy(filename, filename_size, bs->backing_file);

}
