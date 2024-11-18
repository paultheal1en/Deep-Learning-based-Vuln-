QEMUClockType timerlist_get_clock(QEMUTimerList *timer_list)

{

    return timer_list->clock->type;

}
