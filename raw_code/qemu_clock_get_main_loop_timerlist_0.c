QEMUTimerList *qemu_clock_get_main_loop_timerlist(QEMUClockType type)

{

    return main_loop_tlg.tl[type];

}
