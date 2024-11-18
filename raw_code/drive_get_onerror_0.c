BlockInterfaceErrorAction drive_get_onerror(BlockDriverState *bdrv)

{

    DriveInfo *dinfo;



    TAILQ_FOREACH(dinfo, &drives, next) {

        if (dinfo->bdrv == bdrv)

            return dinfo->onerror;

    }



    return BLOCK_ERR_STOP_ENOSPC;

}
