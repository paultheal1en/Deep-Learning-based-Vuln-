static void __attribute__((__constructor__)) rcu_init(void)

{

    QemuThread thread;



    qemu_mutex_init(&rcu_gp_lock);

    qemu_event_init(&rcu_gp_event, true);



    qemu_event_init(&rcu_call_ready_event, false);

    qemu_thread_create(&thread, "call_rcu", call_rcu_thread,

                       NULL, QEMU_THREAD_DETACHED);



    rcu_register_thread();

}
