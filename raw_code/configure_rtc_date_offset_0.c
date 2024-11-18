static void configure_rtc_date_offset(const char *startdate, int legacy)

{

    time_t rtc_start_date;

    struct tm tm;



    if (!strcmp(startdate, "now") && legacy) {

        rtc_date_offset = -1;

    } else {

        if (sscanf(startdate, "%d-%d-%dT%d:%d:%d",

                   &tm.tm_year,

                   &tm.tm_mon,

                   &tm.tm_mday,

                   &tm.tm_hour,

                   &tm.tm_min,

                   &tm.tm_sec) == 6) {

            /* OK */

        } else if (sscanf(startdate, "%d-%d-%d",

                          &tm.tm_year,

                          &tm.tm_mon,

                          &tm.tm_mday) == 3) {

            tm.tm_hour = 0;

            tm.tm_min = 0;

            tm.tm_sec = 0;

        } else {

            goto date_fail;

        }

        tm.tm_year -= 1900;

        tm.tm_mon--;

        rtc_start_date = mktimegm(&tm);

        if (rtc_start_date == -1) {

        date_fail:

            fprintf(stderr, "Invalid date format. Valid formats are:\n"

                            "'2006-06-17T16:01:21' or '2006-06-17'\n");

            exit(1);

        }

        rtc_date_offset = time(NULL) - rtc_start_date;

    }

}
