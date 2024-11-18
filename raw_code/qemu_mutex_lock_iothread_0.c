void qemu_mutex_lock_iothread(void)

{

    if (kvm_enabled()) {

        qemu_mutex_lock(&qemu_fair_mutex);

        qemu_mutex_lock(&qemu_global_mutex);

        qemu_mutex_unlock(&qemu_fair_mutex);

    } else

        qemu_signal_lock(100);

}
