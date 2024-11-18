bool replay_checkpoint(ReplayCheckpoint checkpoint)

{

    bool res = false;

    assert(EVENT_CHECKPOINT + checkpoint <= EVENT_CHECKPOINT_LAST);

    replay_save_instructions();



    if (!replay_file) {

        return true;

    }



    replay_mutex_lock();



    if (replay_mode == REPLAY_MODE_PLAY) {

        if (replay_next_event_is(EVENT_CHECKPOINT + checkpoint)) {

            replay_finish_event();

        } else if (replay_data_kind != EVENT_ASYNC) {

            res = false;

            goto out;

        }

        replay_read_events(checkpoint);

        /* replay_read_events may leave some unread events.

           Return false if not all of the events associated with

           checkpoint were processed */

        res = replay_data_kind != EVENT_ASYNC;

    } else if (replay_mode == REPLAY_MODE_RECORD) {

        replay_put_event(EVENT_CHECKPOINT + checkpoint);

        replay_save_events(checkpoint);

        res = true;

    }

out:

    replay_mutex_unlock();

    return res;

}
