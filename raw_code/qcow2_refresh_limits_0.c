static void qcow2_refresh_limits(BlockDriverState *bs, Error **errp)

{

    BDRVQcow2State *s = bs->opaque;



    if (bs->encrypted) {

        /* Encryption works on a sector granularity */

        bs->request_alignment = BDRV_SECTOR_SIZE;

    }

    bs->bl.pwrite_zeroes_alignment = s->cluster_size;

}
