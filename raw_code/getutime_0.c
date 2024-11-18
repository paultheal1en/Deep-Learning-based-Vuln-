static int64_t getutime(void)

{

#ifdef HAVE_GETRUSAGE

    struct rusage rusage;



    getrusage(RUSAGE_SELF, &rusage);

    return (rusage.ru_utime.tv_sec * 1000000LL) + rusage.ru_utime.tv_usec;

#elif defined(__MINGW32__)

  return av_gettime();

#endif

}
