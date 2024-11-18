void timerlist_notify(QEMUTimerList *timer_list)

{

    if (timer_list->notify_cb) {

        timer_list->notify_cb(timer_list->notify_opaque);

    } else {

        qemu_notify_event();

    }

}
