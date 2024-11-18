void replay_bh_schedule_event(QEMUBH *bh)

{

    if (replay_mode != REPLAY_MODE_NONE) {

        uint64_t id = replay_get_current_step();

        replay_add_event(REPLAY_ASYNC_EVENT_BH, bh, NULL, id);

    } else {

        qemu_bh_schedule(bh);

    }

}
