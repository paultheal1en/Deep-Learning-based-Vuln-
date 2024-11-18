static int qemu_rbd_snap_rollback(BlockDriverState *bs,

                                  const char *snapshot_name)

{

    BDRVRBDState *s = bs->opaque;

    int r;



    r = rbd_snap_rollback(s->image, snapshot_name);

    return r;

}
