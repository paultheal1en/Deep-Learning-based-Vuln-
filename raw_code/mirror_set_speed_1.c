static void mirror_set_speed(BlockJob *job, int64_t speed, Error **errp)

{

    MirrorBlockJob *s = container_of(job, MirrorBlockJob, common);



    if (speed < 0) {

        error_setg(errp, QERR_INVALID_PARAMETER, "speed");

        return;

    }

    ratelimit_set_speed(&s->limit, speed / BDRV_SECTOR_SIZE, SLICE_TIME);

}
