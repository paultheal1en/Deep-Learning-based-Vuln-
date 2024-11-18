static void *clone_func(void *arg)

{

    new_thread_info *info = arg;

    CPUArchState *env;

    CPUState *cpu;

    TaskState *ts;



    rcu_register_thread();

    env = info->env;

    cpu = ENV_GET_CPU(env);

    thread_cpu = cpu;

    ts = (TaskState *)cpu->opaque;

    info->tid = gettid();

    cpu->host_tid = info->tid;

    task_settid(ts);

    if (info->child_tidptr)

        put_user_u32(info->tid, info->child_tidptr);

    if (info->parent_tidptr)

        put_user_u32(info->tid, info->parent_tidptr);

    /* Enable signals.  */

    sigprocmask(SIG_SETMASK, &info->sigmask, NULL);

    /* Signal to the parent that we're ready.  */

    pthread_mutex_lock(&info->mutex);

    pthread_cond_broadcast(&info->cond);

    pthread_mutex_unlock(&info->mutex);

    /* Wait until the parent has finshed initializing the tls state.  */

    pthread_mutex_lock(&clone_lock);

    pthread_mutex_unlock(&clone_lock);

    cpu_loop(env);

    /* never exits */

    return NULL;

}
