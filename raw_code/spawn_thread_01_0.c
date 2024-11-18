static void spawn_thread(ThreadPool *pool)

{

    pool->cur_threads++;

    pool->new_threads++;

    /* If there are threads being created, they will spawn new workers, so

     * we don't spend time creating many threads in a loop holding a mutex or

     * starving the current vcpu.

     *

     * If there are no idle threads, ask the main thread to create one, so we

     * inherit the correct affinity instead of the vcpu affinity.

     */

    if (!pool->pending_threads) {

        qemu_bh_schedule(pool->new_thread_bh);

    }

}
