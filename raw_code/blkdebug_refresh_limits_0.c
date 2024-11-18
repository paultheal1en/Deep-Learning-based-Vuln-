static void blkdebug_refresh_limits(BlockDriverState *bs, Error **errp)

{

    BDRVBlkdebugState *s = bs->opaque;



    if (s->align) {

        bs->request_alignment = s->align;

    }

}
