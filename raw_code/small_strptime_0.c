const char *small_strptime(const char *p, const char *fmt,

                           struct tm *dt)

{

    int c, val;



    for(;;) {

        c = *fmt++;

        if (c == '\0') {

            return p;

        } else if (c == '%') {

            c = *fmt++;

            switch(c) {

            case 'H':

                val = date_get_num(&p, 0, 23, 2);

                if (val == -1)

                    return NULL;

                dt->tm_hour = val;

                break;

            case 'M':

                val = date_get_num(&p, 0, 59, 2);

                if (val == -1)

                    return NULL;

                dt->tm_min = val;

                break;

            case 'S':

                val = date_get_num(&p, 0, 59, 2);

                if (val == -1)

                    return NULL;

                dt->tm_sec = val;

                break;

            case 'Y':

                val = date_get_num(&p, 0, 9999, 4);

                if (val == -1)

                    return NULL;

                dt->tm_year = val - 1900;

                break;

            case 'm':

                val = date_get_num(&p, 1, 12, 2);

                if (val == -1)

                    return NULL;

                dt->tm_mon = val - 1;

                break;

            case 'd':

                val = date_get_num(&p, 1, 31, 2);

                if (val == -1)

                    return NULL;

                dt->tm_mday = val;

                break;

            case '%':

                goto match;

            default:

                return NULL;

            }

        } else {

        match:

            if (c != *p)

                return NULL;

            p++;

        }

    }

    return p;

}
