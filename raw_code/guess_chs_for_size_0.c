static void guess_chs_for_size(BlockDriverState *bs,

                uint32_t *pcyls, uint32_t *pheads, uint32_t *psecs)

{

    uint64_t nb_sectors;

    int cylinders;



    bdrv_get_geometry(bs, &nb_sectors);



    cylinders = nb_sectors / (16 * 63);

    if (cylinders > 16383) {

        cylinders = 16383;

    } else if (cylinders < 2) {

        cylinders = 2;

    }

    *pcyls = cylinders;

    *pheads = 16;

    *psecs = 63;

}
