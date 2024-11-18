static void *migration_thread(void *opaque)

{

    MigrationState *s = opaque;

    /* Used by the bandwidth calcs, updated later */

    int64_t initial_time = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

    int64_t setup_start = qemu_clock_get_ms(QEMU_CLOCK_HOST);

    int64_t initial_bytes = 0;

    int64_t max_size = 0;

    int64_t start_time = initial_time;

    int64_t end_time;

    bool old_vm_running = false;

    bool entered_postcopy = false;

    /* The active state we expect to be in; ACTIVE or POSTCOPY_ACTIVE */

    enum MigrationStatus current_active_state = MIGRATION_STATUS_ACTIVE;



    rcu_register_thread();



    qemu_savevm_state_header(s->to_dst_file);



    if (migrate_postcopy_ram()) {

        /* Now tell the dest that it should open its end so it can reply */

        qemu_savevm_send_open_return_path(s->to_dst_file);



        /* And do a ping that will make stuff easier to debug */

        qemu_savevm_send_ping(s->to_dst_file, 1);



        /*

         * Tell the destination that we *might* want to do postcopy later;

         * if the other end can't do postcopy it should fail now, nice and

         * early.

         */

        qemu_savevm_send_postcopy_advise(s->to_dst_file);

    }



    qemu_savevm_state_begin(s->to_dst_file, &s->params);



    s->setup_time = qemu_clock_get_ms(QEMU_CLOCK_HOST) - setup_start;

    current_active_state = MIGRATION_STATUS_ACTIVE;

    migrate_set_state(&s->state, MIGRATION_STATUS_SETUP,

                      MIGRATION_STATUS_ACTIVE);



    trace_migration_thread_setup_complete();



    while (s->state == MIGRATION_STATUS_ACTIVE ||

           s->state == MIGRATION_STATUS_POSTCOPY_ACTIVE) {

        int64_t current_time;

        uint64_t pending_size;



        if (!qemu_file_rate_limit(s->to_dst_file)) {

            uint64_t pend_post, pend_nonpost;



            qemu_savevm_state_pending(s->to_dst_file, max_size, &pend_nonpost,

                                      &pend_post);

            pending_size = pend_nonpost + pend_post;

            trace_migrate_pending(pending_size, max_size,

                                  pend_post, pend_nonpost);

            if (pending_size && pending_size >= max_size) {

                /* Still a significant amount to transfer */



                if (migrate_postcopy_ram() &&

                    s->state != MIGRATION_STATUS_POSTCOPY_ACTIVE &&

                    pend_nonpost <= max_size &&

                    atomic_read(&s->start_postcopy)) {



                    if (!postcopy_start(s, &old_vm_running)) {

                        current_active_state = MIGRATION_STATUS_POSTCOPY_ACTIVE;

                        entered_postcopy = true;

                    }



                    continue;

                }

                /* Just another iteration step */

                qemu_savevm_state_iterate(s->to_dst_file, entered_postcopy);

            } else {

                trace_migration_thread_low_pending(pending_size);

                migration_completion(s, current_active_state,

                                     &old_vm_running, &start_time);

                break;

            }

        }



        if (qemu_file_get_error(s->to_dst_file)) {

            migrate_set_state(&s->state, current_active_state,

                              MIGRATION_STATUS_FAILED);

            trace_migration_thread_file_err();

            break;

        }

        current_time = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

        if (current_time >= initial_time + BUFFER_DELAY) {

            uint64_t transferred_bytes = qemu_ftell(s->to_dst_file) -

                                         initial_bytes;

            uint64_t time_spent = current_time - initial_time;

            double bandwidth = (double)transferred_bytes / time_spent;

            max_size = bandwidth * migrate_max_downtime() / 1000000;



            s->mbps = (((double) transferred_bytes * 8.0) /

                    ((double) time_spent / 1000.0)) / 1000.0 / 1000.0;



            trace_migrate_transferred(transferred_bytes, time_spent,

                                      bandwidth, max_size);

            /* if we haven't sent anything, we don't want to recalculate

               10000 is a small enough number for our purposes */

            if (s->dirty_bytes_rate && transferred_bytes > 10000) {

                s->expected_downtime = s->dirty_bytes_rate / bandwidth;

            }



            qemu_file_reset_rate_limit(s->to_dst_file);

            initial_time = current_time;

            initial_bytes = qemu_ftell(s->to_dst_file);

        }

        if (qemu_file_rate_limit(s->to_dst_file)) {

            /* usleep expects microseconds */

            g_usleep((initial_time + BUFFER_DELAY - current_time)*1000);

        }

    }



    trace_migration_thread_after_loop();

    /* If we enabled cpu throttling for auto-converge, turn it off. */

    cpu_throttle_stop();

    end_time = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);



    qemu_mutex_lock_iothread();

    qemu_savevm_state_cleanup();

    if (s->state == MIGRATION_STATUS_COMPLETED) {

        uint64_t transferred_bytes = qemu_ftell(s->to_dst_file);

        s->total_time = end_time - s->total_time;

        if (!entered_postcopy) {

            s->downtime = end_time - start_time;

        }

        if (s->total_time) {

            s->mbps = (((double) transferred_bytes * 8.0) /

                       ((double) s->total_time)) / 1000;

        }

        runstate_set(RUN_STATE_POSTMIGRATE);

    } else {

        if (old_vm_running && !entered_postcopy) {

            vm_start();

        } else {

            if (runstate_check(RUN_STATE_FINISH_MIGRATE)) {

                runstate_set(RUN_STATE_POSTMIGRATE);

            }

        }

    }

    qemu_bh_schedule(s->cleanup_bh);

    qemu_mutex_unlock_iothread();



    rcu_unregister_thread();

    return NULL;

}
