static void qemu_announce_self_once(void *opaque)

{

    static int count = SELF_ANNOUNCE_ROUNDS;

    QEMUTimer *timer = *(QEMUTimer **)opaque;



    qemu_foreach_nic(qemu_announce_self_iter, NULL);



    if (--count) {

        /* delay 50ms, 150ms, 250ms, ... */

        qemu_mod_timer(timer, qemu_get_clock(rt_clock) +

                       50 + (SELF_ANNOUNCE_ROUNDS - count - 1) * 100);

    } else {

	    qemu_del_timer(timer);

	    qemu_free_timer(timer);

    }

}
