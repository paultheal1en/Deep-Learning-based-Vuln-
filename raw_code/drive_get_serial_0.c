const char *drive_get_serial(BlockDriverState *bdrv)

{

    DriveInfo *dinfo;



    TAILQ_FOREACH(dinfo, &drives, next) {

        if (dinfo->bdrv == bdrv)

            return dinfo->serial;

    }



    return "\0";

}
