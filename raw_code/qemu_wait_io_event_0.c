static void qemu_wait_io_event(CPUState *env)

{

    while (!tcg_has_work())

        qemu_cond_timedwait(env->halt_cond, &qemu_global_mutex, 1000);



    qemu_mutex_unlock(&qemu_global_mutex);



    /*

     * Users of qemu_global_mutex can be starved, having no chance

     * to acquire it since this path will get to it first.

     * So use another lock to provide fairness.

     */

    qemu_mutex_lock(&qemu_fair_mutex);

    qemu_mutex_unlock(&qemu_fair_mutex);



    qemu_mutex_lock(&qemu_global_mutex);

    qemu_wait_io_event_common(env);

}
