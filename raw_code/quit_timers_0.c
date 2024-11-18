static void quit_timers(void)

{

    alarm_timer->stop(alarm_timer);

    alarm_timer = NULL;

}
