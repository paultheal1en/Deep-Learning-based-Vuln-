void bdrv_get_geometry(BlockDriverState *bs, uint64_t *nb_sectors_ptr)

{

    int64_t nb_sectors = bdrv_nb_sectors(bs);



    *nb_sectors_ptr = nb_sectors < 0 ? 0 : nb_sectors;

}
