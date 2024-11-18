int bdrv_snapshot_create(BlockDriverState *bs,

                         QEMUSnapshotInfo *sn_info)

{

    BlockDriver *drv = bs->drv;

    if (!drv)

        return -ENOMEDIUM;

    if (drv->bdrv_snapshot_create)

        return drv->bdrv_snapshot_create(bs, sn_info);

    if (bs->file)

        return bdrv_snapshot_create(bs->file, sn_info);

    return -ENOTSUP;

}
