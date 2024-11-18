bool block_job_user_paused(BlockJob *job)

{

    return job ? job->user_paused : 0;

}
