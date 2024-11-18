static void win32_rearm_timer(struct qemu_alarm_timer *t,

                              int64_t nearest_delta_ns)

{

    HANDLE hTimer = t->timer;

    int nearest_delta_ms;

    BOOLEAN success;



    nearest_delta_ms = (nearest_delta_ns + 999999) / 1000000;

    if (nearest_delta_ms < 1) {

        nearest_delta_ms = 1;

    }

    success = ChangeTimerQueueTimer(NULL,

                                    hTimer,

                                    nearest_delta_ms,

                                    3600000);



    if (!success) {

        fprintf(stderr, "Failed to rearm win32 alarm timer: %ld\n",

                GetLastError());

        exit(-1);

    }



}
