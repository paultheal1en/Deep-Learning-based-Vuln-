void drive_uninit(BlockDriverState *bdrv)

{

    DriveInfo *dinfo;



    TAILQ_FOREACH(dinfo, &drives, next) {

        if (dinfo->bdrv != bdrv)

            continue;

        qemu_opts_del(dinfo->opts);

        TAILQ_REMOVE(&drives, dinfo, next);

        qemu_free(dinfo);

        break;

    }

}
