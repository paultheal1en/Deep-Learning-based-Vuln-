static void qemu_clock_init(QEMUClockType type)

{

    QEMUClock *clock = qemu_clock_ptr(type);



    /* Assert that the clock of type TYPE has not been initialized yet. */

    assert(main_loop_tlg.tl[type] == NULL);



    clock->type = type;

    clock->enabled = (type == QEMU_CLOCK_VIRTUAL ? false : true);

    clock->last = INT64_MIN;

    QLIST_INIT(&clock->timerlists);

    notifier_list_init(&clock->reset_notifiers);

    main_loop_tlg.tl[type] = timerlist_new(type, NULL, NULL);

}
