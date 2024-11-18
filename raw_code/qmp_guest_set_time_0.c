void qmp_guest_set_time(bool has_time, int64_t time_ns, Error **errp)

{

    Error *local_err = NULL;

    SYSTEMTIME ts;

    FILETIME tf;

    LONGLONG time;



    if (!has_time) {

        /* Unfortunately, Windows libraries don't provide an easy way to access

         * RTC yet:

         *

         * https://msdn.microsoft.com/en-us/library/aa908981.aspx

         */

        error_setg(errp, "Time argument is required on this platform");

        return;

    }



    /* Validate time passed by user. */

    if (time_ns < 0 || time_ns / 100 > INT64_MAX - W32_FT_OFFSET) {

        error_setg(errp, "Time %" PRId64 "is invalid", time_ns);

        return;

    }



    time = time_ns / 100 + W32_FT_OFFSET;



    tf.dwLowDateTime = (DWORD) time;

    tf.dwHighDateTime = (DWORD) (time >> 32);



    if (!FileTimeToSystemTime(&tf, &ts)) {

        error_setg(errp, "Failed to convert system time %d",

                   (int)GetLastError());

        return;

    }



    acquire_privilege(SE_SYSTEMTIME_NAME, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    if (!SetSystemTime(&ts)) {

        error_setg(errp, "Failed to set time to guest: %d", (int)GetLastError());

        return;

    }

}
