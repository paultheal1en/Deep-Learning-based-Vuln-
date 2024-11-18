static int stream_set_speed(BlockJob *job, int64_t value)

{

    StreamBlockJob *s = container_of(job, StreamBlockJob, common);



    if (value < 0) {

        return -EINVAL;

    }

    ratelimit_set_speed(&s->limit, value / BDRV_SECTOR_SIZE);

    return 0;

}
