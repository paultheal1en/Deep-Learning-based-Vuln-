bool timer_expired(QEMUTimer *timer_head, int64_t current_time)

{

    return timer_expired_ns(timer_head, current_time * timer_head->scale);

}
