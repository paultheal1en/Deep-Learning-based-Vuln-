static int cdrom_open(BlockDriverState *bs, QDict *options, int flags,

                      Error **errp)

{

    BDRVRawState *s = bs->opaque;

    int ret;



    s->type = FTYPE_CD;



    /* open will not fail even if no CD is inserted, so add O_NONBLOCK */

    ret = raw_open_common(bs, options, flags, O_NONBLOCK, errp);

    return ret;

}
