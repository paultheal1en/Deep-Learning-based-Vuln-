bool timerlist_has_timers(QEMUTimerList *timer_list)

{

    return !!timer_list->active_timers;

}
