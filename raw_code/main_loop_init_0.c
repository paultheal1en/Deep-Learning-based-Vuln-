int main_loop_init(void)

{

    int ret;



    qemu_mutex_lock_iothread();

    ret = qemu_signal_init();

    if (ret) {

        return ret;

    }



    /* Note eventfd must be drained before signalfd handlers run */

    ret = qemu_event_init();

    if (ret) {

        return ret;

    }



    return 0;

}
