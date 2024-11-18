int block_job_set_speed(BlockJob *job, int64_t value)

{

    int rc;



    if (!job->job_type->set_speed) {

        return -ENOTSUP;

    }

    rc = job->job_type->set_speed(job, value);

    if (rc == 0) {

        job->speed = value;

    }

    return rc;

}
