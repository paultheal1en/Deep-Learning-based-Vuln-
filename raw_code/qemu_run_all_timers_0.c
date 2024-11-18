void qemu_run_all_timers(void)

{

    alarm_timer->pending = 0;



    /* rearm timer, if not periodic */

    if (alarm_timer->expired) {

        alarm_timer->expired = 0;

        qemu_rearm_alarm_timer(alarm_timer);

    }



    /* vm time timers */

    qemu_run_timers(vm_clock);

    qemu_run_timers(rt_clock);

    qemu_run_timers(host_clock);

}
