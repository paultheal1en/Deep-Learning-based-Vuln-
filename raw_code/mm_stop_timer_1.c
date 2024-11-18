static void mm_stop_timer(struct qemu_alarm_timer *t)

{

    timeKillEvent(mm_timer);

    timeEndPeriod(mm_period);

}
