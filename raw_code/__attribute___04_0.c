static void __attribute__((constructor)) coroutine_pool_init(void)

{

    qemu_mutex_init(&pool_lock);

}
