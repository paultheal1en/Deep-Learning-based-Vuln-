static int64_t get_clock(void)

{

#if defined(__linux__) || (defined(__FreeBSD__) && __FreeBSD_version >= 500000) \

	|| defined(__DragonFly__)

    if (use_rt_clock) {

        struct timespec ts;

        clock_gettime(CLOCK_MONOTONIC, &ts);

        return ts.tv_sec * 1000000000LL + ts.tv_nsec;

    } else

#endif

    {

        /* XXX: using gettimeofday leads to problems if the date

           changes, so it should be avoided. */

        struct timeval tv;

        gettimeofday(&tv, NULL);

        return tv.tv_sec * 1000000000LL + (tv.tv_usec * 1000);

    }

}
