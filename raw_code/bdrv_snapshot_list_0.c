int bdrv_snapshot_list(BlockDriverState *bs,

                       QEMUSnapshotInfo **psn_info)

{

    BlockDriver *drv = bs->drv;

    if (!drv)

        return -ENOMEDIUM;

    if (drv->bdrv_snapshot_list)

        return drv->bdrv_snapshot_list(bs, psn_info);

    if (bs->file)

        return bdrv_snapshot_list(bs->file, psn_info);

    return -ENOTSUP;

}
