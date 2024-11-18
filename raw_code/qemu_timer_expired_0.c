int qemu_timer_expired(QEMUTimer *timer_head, int64_t current_time)

{

    if (!timer_head)

        return 0;

    return (timer_head->expire_time <= current_time);

}
